//
//  VoxCPM2Model.swift
//  MLXAudio
//
//  VoxCPM2 TTS model: 2B params, 48kHz, 30 languages.
//  Autoregressive MiniCPM backbone + CFM diffusion + AudioVAE.
//  Ported from mlx-audio Python: voxcpm2/voxcpm2.py
//

import Accelerate
import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXAudioCore
import Tokenizers

// MARK: - Reference resampling

/// The resampler the Python port encodes references with (`mlx_audio.resample.resample_audio_array`):
/// scipy `resample_poly` with resampy's `kaiser_best` FIR - `firwin(2 * 64 * max_rate + 1,
/// 0.9475937167399596 / max_rate, window=("kaiser", 14.769656459379492))`, `padtype="edge"`,
/// computed in float64. Only the integer decimation case (up = 1) is ported.
public enum KaiserBestResampler {
    static let numZeros = 64
    static let rolloff = 0.9475937167399596
    static let beta = 14.769656459379492

    /// nil when `from / to` is not an integer.
    public static func decimate(_ x: [Float], from: Int, to: Int) -> [Float]? {
        guard from > to, from % to == 0, !x.isEmpty else { return nil }
        let down = from / to
        let h = firwin(numTaps: 2 * numZeros * down + 1, cutoff: rolloff / Double(down))
        let halfLength = (h.count - 1) / 2
        let outCount = (x.count + down - 1) / down

        // y[n] = sum_j h[j] * x[down*n + halfLength - j], x extended by its edge values
        // (scipy upfirdn, mode "edge"). vDSP_desampD computes C[n] = sum_p A[down*n + p] * F[p],
        // so A is x shifted by halfLength and F is h reversed.
        let padded = (0 ..< (down * (outCount - 1) + h.count)).map { i -> Double in
            Double(x[min(max(i - halfLength, 0), x.count - 1)])
        }
        let reversed = Array(h.reversed())
        var y = [Double](repeating: 0, count: outCount)
        vDSP_desampD(padded, vDSP_Stride(down), reversed, &y, vDSP_Length(outCount), vDSP_Length(h.count))
        return y.map { Float($0) }
    }

    /// scipy.signal.firwin, low-pass, Kaiser window, scaled to unit gain at DC.
    static func firwin(numTaps: Int, cutoff: Double) -> [Double] {
        let alpha = Double(numTaps - 1) / 2
        let i0Beta = besselI0(beta)
        var h = (0 ..< numTaps).map { n -> Double in
            let m = Double(n) - alpha
            let x = cutoff * m
            let sinc = x == 0 ? 1 : Foundation.sin(Double.pi * x) / (Double.pi * x)
            let r = Double(n) / alpha - 1
            return cutoff * sinc * besselI0(beta * (1 - r * r).squareRoot()) / i0Beta
        }
        let sum = h.reduce(0, +)
        for n in h.indices { h[n] /= sum }
        return h
    }

    static func besselI0(_ x: Double) -> Double {
        var sum = 1.0
        var term = 1.0
        let q = x * x / 4
        for k in 1 ..< 500 {
            term *= q / Double(k * k)
            sum += term
            if term < sum * 1e-17 { break }
        }
        return sum
    }
}

// MARK: - Scalar Quantization

class ScalarQuantizationLayer: Module {
    let scale: Int
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(inDim: Int, outDim: Int, latentDim: Int = 64, scale: Int = 9) {
        self.scale = scale
        self._inProj.wrappedValue = Linear(inDim, latentDim)
        self._outProj.wrappedValue = Linear(latentDim, outDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inProj(x)
        h = MLX.tanh(h)
        h = MLX.round(h * Float(scale)) / Float(scale)
        return outProj(h)
    }
}

// MARK: - VoxCPM2 Model

public final class VoxCPM2Model: Module, SpeechGenerationModel, @unchecked Sendable {

    // MARK: - Configuration

    public let config: VoxCPM2Configuration
    let patchSize: Int
    let featDim: Int

    // MARK: - Sub-models

    @ModuleInfo(key: "base_lm") var baseLM: VoxMiniCPMModel
    @ModuleInfo(key: "residual_lm") var residualLM: VoxMiniCPMModel
    @ModuleInfo(key: "feat_encoder") var featEncoder: VoxCPMLocEnc
    @ModuleInfo(key: "feat_decoder") var featDecoder: VoxUnifiedCFM

    @ModuleInfo(key: "fsq_layer") var fsqLayer: ScalarQuantizationLayer
    @ModuleInfo(key: "enc_to_lm_proj") var encToLmProj: Linear
    @ModuleInfo(key: "lm_to_dit_proj") var lmToDitProj: Linear
    @ModuleInfo(key: "res_to_dit_proj") var resToDitProj: Linear
    @ModuleInfo(key: "fusion_concat_proj") var fusionConcatProj: Linear

    @ModuleInfo(key: "stop_proj") var stopProj: Linear
    @ModuleInfo(key: "stop_head") var stopHead: Linear

    @ModuleInfo(key: "audio_vae") var audioVAE: VoxAudioVAE

    // MARK: - State

    public let tokenizer: Tokenizers.Tokenizer?

    // MARK: - Special tokens (defined in VoxCPM2 tokenizer config, ids 101-104)

    let audioStartToken: Int32 = 101
    let audioEndToken: Int32 = 102
    let refAudioStartToken: Int32 = 103
    let refAudioEndToken: Int32 = 104

    // MARK: - Protocol conformance

    public var sampleRate: Int { config.audioVaeConfig.outSampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: 1.0)
    }

    // MARK: - Initialization

    public init(_ config: VoxCPM2Configuration, tokenizer: Tokenizers.Tokenizer? = nil) {
        self.config = config
        self.tokenizer = tokenizer
        self.patchSize = config.patchSize
        self.featDim = config.featDim

        let lmConfig = config.lmConfig

        // Base LM (full layers, with vocab)
        self._baseLM.wrappedValue = VoxMiniCPMModel(lmConfig)

        // Residual LM (fewer layers, vocab_size=0, optionally no_rope)
        var resConfig = lmConfig
        resConfig.numHiddenLayers = config.residualLmNumLayers
        resConfig.vocabSize = 0
        resConfig.noRope = config.residualLmNoRope
        self._residualLM.wrappedValue = VoxMiniCPMModel(resConfig)

        // Encoder
        var encConfig = lmConfig
        encConfig.hiddenSize = config.encoderConfig.hiddenDim
        encConfig.intermediateSize = config.encoderConfig.ffnDim
        encConfig.numAttentionHeads = config.encoderConfig.numHeads
        encConfig.numHiddenLayers = config.encoderConfig.numLayers
        encConfig.kvChannels = config.encoderConfig.kvChannels
        encConfig.vocabSize = 0
        self._featEncoder.wrappedValue = VoxCPMLocEnc(encConfig, inputDim: config.featDim)

        // DiT / CFM
        var ditConfig = lmConfig
        ditConfig.hiddenSize = config.ditConfig.hiddenDim
        ditConfig.intermediateSize = config.ditConfig.ffnDim
        ditConfig.numAttentionHeads = config.ditConfig.numHeads
        ditConfig.numHiddenLayers = config.ditConfig.numLayers
        ditConfig.kvChannels = config.ditConfig.kvChannels
        ditConfig.vocabSize = 0

        let estimator = VoxCPMLocDiT(ditConfig, inChannels: config.featDim)
        self._featDecoder.wrappedValue = VoxUnifiedCFM(
            inChannels: config.featDim,
            cfmParams: config.ditConfig.cfmConfig,
            estimator: estimator,
            meanMode: config.ditConfig.ditMeanMode
        )

        // Projections
        self._fsqLayer.wrappedValue = ScalarQuantizationLayer(
            inDim: lmConfig.hiddenSize,
            outDim: lmConfig.hiddenSize,
            latentDim: config.scalarQuantizationLatentDim,
            scale: config.scalarQuantizationScale
        )

        self._encToLmProj.wrappedValue = Linear(
            config.encoderConfig.hiddenDim, lmConfig.hiddenSize
        )
        self._lmToDitProj.wrappedValue = Linear(
            lmConfig.hiddenSize, config.ditConfig.hiddenDim
        )
        self._resToDitProj.wrappedValue = Linear(
            lmConfig.hiddenSize, config.ditConfig.hiddenDim
        )

        // V2: fusion_concat_proj
        self._fusionConcatProj.wrappedValue = Linear(
            lmConfig.hiddenSize * 2, lmConfig.hiddenSize
        )

        // Stop predictor
        self._stopProj.wrappedValue = Linear(lmConfig.hiddenSize, lmConfig.hiddenSize)
        self._stopHead.wrappedValue = Linear(lmConfig.hiddenSize, 2, bias: false)

        // Audio VAE
        self._audioVAE.wrappedValue = VoxAudioVAE(config.audioVaeConfig)

        super.init()
    }

    // MARK: - Audio Encoding

    /// Which side `encodeWav` pads to patch alignment: the reference on the right, the Hi-Fi
    /// prompt on the left, as Python's `_encode_wav(padding_mode:)`.
    public enum Padding: Sendable { case right, left }

    public func encodeWav(_ audio: MLXArray, padding: Padding = .right) throws -> MLXArray {
        var inp = audio

        // Resample from output rate (48kHz) to encoder rate (16kHz) if needed
        let outRate = sampleRate
        let encRate = audioVAE.encodeSampleRate
        if outRate != encRate {
            let flat = inp.ndim == 1 ? inp : inp.flattened()
            let samples = flat.asArray(Float.self)
            // Python's resampler where the ratio is an integer decimation (48 -> 16 kHz): with
            // AVAudioConverter's defaults the reference latents differed from Python's by 13 %.
            let resampled = try KaiserBestResampler.decimate(samples, from: outRate, to: encRate)
                ?? resampleAudio(samples, from: outRate, to: encRate)
            inp = MLXArray(resampled)
        }

        if inp.ndim == 1 {
            inp = inp.expandedDimensions(axes: [0, 1]) // (1, 1, T)
        } else if inp.ndim == 2 {
            inp = inp.expandedDimensions(axis: 1) // (1, 1, T) or (B, 1, T)
        }

        // Pad to patch alignment
        let patchLen = patchSize * audioVAE.chunkSize
        let length = inp.dim(-1)
        let remainder = length % patchLen
        if remainder != 0 {
            let padSize = patchLen - remainder
            let side = padding == .left ? IntOrPair((padSize, 0)) : IntOrPair((0, padSize))
            inp = MLX.padded(inp, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), side])
        }

        // VAE encode: input is (B, 1, T) in channel-first
        // CausalEncoder expects (B, T, C) in channel-last
        let channelLast = inp.transposed(0, 2, 1) // (B, T, 1)
        let feat = audioVAE.encode(channelLast, sampleRate: audioVAE.encodeSampleRate)
        // feat: (B, T', D) in channel-last

        let squeezed = feat.squeezed(axis: 0) // (T', D)

        // Reshape into patches: (T', D) → (numPatches, patchSize, D)
        let tPrime = squeezed.dim(0)
        let numPatches = tPrime / patchSize
        let trimmed = squeezed[..<(numPatches * patchSize)]
        return trimmed.reshaped(numPatches, patchSize, -1)
    }

    func makeRefPrefix(_ refFeat: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let refLen = refFeat.dim(0)
        let z1 = MLXArray.zeros([1, patchSize, featDim])

        let tokens = MLX.concatenated([
            MLXArray([refAudioStartToken]),
            MLXArray.zeros([refLen]).asType(.int32),
            MLXArray([refAudioEndToken]),
        ])

        let feats = MLX.concatenated([z1, refFeat, z1], axis: 0)

        let tMask = MLX.concatenated([
            MLXArray([Float(1)]),
            MLXArray.zeros([refLen]),
            MLXArray([Float(1)]),
        ])
        let aMask = MLX.concatenated([
            MLXArray([Float(0)]),
            MLXArray.ones([refLen]),
            MLXArray([Float(0)]),
        ])

        return (tokens, feats, tMask, aMask)
    }

    // MARK: - Tokenization

    public func tokenize(_ text: String) throws -> [Int] {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        // Match Python: tokenize + convert_tokens_to_ids (no BOS/EOS)
        var ids = tokenizer.encode(text: text).map { Int($0) }
        if let first = ids.first, first == (tokenizer.bosTokenId ?? -1) {
            ids.removeFirst()
        }
        if let last = ids.last, last == (tokenizer.eosTokenId ?? -1) {
            ids.removeLast()
        }
        return ids.flatMap { Self.singleCharacterChineseIDs(for: $0, tokenizer: tokenizer) }
    }

    static func singleCharacterChineseIDs(for id: Int, tokenizer: Tokenizers.Tokenizer) -> [Int] {
        guard let token = tokenizer.convertIdToToken(id) else { return [id] }
        let word = token.replacingOccurrences(of: "▁", with: "")
        let characters = word.unicodeScalars.map { String($0) }
        guard characters.count >= 2,
            word.unicodeScalars.allSatisfy({ (0x4E00 ... 0x9FFF).contains($0.value) }),
            Self.vocabularyID(of: word, tokenizer: tokenizer) != nil
        else { return [id] }
        let ids = characters.compactMap { Self.vocabularyID(of: $0, tokenizer: tokenizer) }
        return ids.count == characters.count ? ids : [id]
    }

    private static func vocabularyID(of token: String, tokenizer: Tokenizers.Tokenizer) -> Int? {
        guard let id = tokenizer.convertTokenToId(token), id != tokenizer.unknownTokenId else { return nil }
        return id
    }

    // MARK: - Generation

    /// Every mode of the Python port's `generate` (mlx_audio 0.6.6, `voxcpm2.py`), with cfg and
    /// diffusion steps given per call rather than read from the config:
    /// - `refAudio` + `promptAudio` + `promptText`: Hi-Fi («ultimate cloning», Python mode 4)
    /// - `refAudio`: reference cloning (mode 3)
    /// - `promptAudio` + `promptText`: continuation (mode 2)
    /// - none: zero-shot (mode 1)
    /// Audio is mono at `sampleRate` (48 kHz). There is no instruction parameter: OpenBMB disables
    /// it in ultimate cloning, and the mlx port, which prepends `(instruct)` to the text, gets it
    /// read aloud.
    public func generateAudio(
        text: String,
        refAudio: MLXArray? = nil,
        promptAudio: MLXArray? = nil,
        promptText: String? = nil,
        cfgValue: Float,
        inferenceTimesteps: Int,
        maxTokens: Int = 2000,
        minTokens: Int = 2,
        streamingPrefixLen: Int = 4
    ) throws -> MLXArray {
        let scaleEmb = Float(config.lmConfig.useMup ? config.lmConfig.scaleEmb : 1)
        let latentDim = audioVAE.latentDim

        // The prompt's transcript goes in front of the text, its audio after it (Python modes 2, 4).
        var prompt: (audio: MLXArray, text: String)?
        if let promptAudio, let promptText { prompt = (promptAudio, promptText) }

        let textIds = try tokenize((prompt?.text ?? "") + text)
        let tokenArray = MLXArray((textIds + [Int(audioStartToken)]).map { Int32($0) })
        let textLength = tokenArray.dim(0)

        var tokens: [MLXArray] = []
        var feats: [MLXArray] = []
        var tMasks: [MLXArray] = []
        var aMasks: [MLXArray] = []

        if let refAudio {
            let (refTokens, refFeats, refTMask, refAMask) = makeRefPrefix(try encodeWav(refAudio, padding: .right))
            tokens.append(refTokens)
            feats.append(refFeats)
            tMasks.append(refTMask)
            aMasks.append(refAMask)
        }

        tokens.append(tokenArray)
        feats.append(MLXArray.zeros([textLength, patchSize, latentDim]))
        tMasks.append(MLXArray.ones([textLength]))
        aMasks.append(MLXArray.zeros([textLength]))

        if let prompt {
            let promptFeat = try encodeWav(prompt.audio, padding: .left)
            let promptLength = promptFeat.dim(0)
            tokens.append(MLXArray.zeros([promptLength]).asType(.int32))
            feats.append(promptFeat)
            tMasks.append(MLXArray.zeros([promptLength]))
            aMasks.append(MLXArray.ones([promptLength]))
        }

        let textToken = MLX.concatenated(tokens)
        let audioFeat = MLX.concatenated(feats, axis: 0)
        let textMask = MLX.concatenated(tMasks)
        let audioMask = MLX.concatenated(aMasks)

        // Add batch dimension
        let bTextToken = textToken.expandedDimensions(axis: 0)
        let bAudioFeat = audioFeat.expandedDimensions(axis: 0)

        // Text embedding with scale
        let textEmbed = baseLM.embedTokens!(bTextToken) * scaleEmb

        let dtype = textEmbed.dtype
        let bTextMask = textMask.expandedDimensions(axis: 0).asType(dtype)
        let bAudioMask = audioMask.expandedDimensions(axis: 0).asType(dtype)

        // Encode audio features
        var featEmbed = featEncoder(bAudioFeat.asType(dtype))
        featEmbed = encToLmProj(featEmbed)

        // Combine text and audio embeddings
        let combinedEmbed = bTextMask[0..., 0..., .newAxis] * textEmbed
            + bAudioMask[0..., 0..., .newAxis] * featEmbed

        var prefixFeatCond = bAudioFeat[0..., -1, 0..., 0...] // (1, P, D)

        // Initial forward pass
        var (encOutputs, lmCache) = baseLM(inputsEmbeds: combinedEmbed)

        // Apply FSQ to audio positions
        encOutputs = fsqLayer(encOutputs) * bAudioMask[0..., 0..., .newAxis]
            + encOutputs * bTextMask[0..., 0..., .newAxis]

        var lmHidden = encOutputs[0..., -1, 0...] // (1, H)

        // V2: fusion_concat_proj for residual input
        let residualInput = fusionConcatProj(
            MLX.concatenated([encOutputs, bAudioMask[0..., 0..., .newAxis] * featEmbed], axis: -1)
        )

        var (residualOutputs, resCache) = residualLM(inputsEmbeds: residualInput)
        var residualHidden = residualOutputs[0..., -1, 0...]

        var predFeatSeq: [MLXArray] = []
        predFeatSeq.reserveCapacity(maxTokens + streamingPrefixLen)

        // Continuation (the sequence ends in audio): seed the decoded sequence with the last
        // patches of the prompt, and trim their audio off after decoding - Python's
        // `has_continuation` branch.
        let audioIndices = audioMask.asArray(Float.self).enumerated().filter { $0.element > 0 }.map(\.offset)
        let hasContinuation = audioMask[-1].item(Float.self) == 1
        if hasContinuation {
            let contextLength = min(streamingPrefixLen - 1, audioIndices.count)
            for index in audioIndices.suffix(contextLength) {
                predFeatSeq.append(bAudioFeat[0..., index, 0..., 0...][0..., .newAxis, 0..., 0...])
            }
        }

        // Generation loop
        for i in 0 ..< maxTokens {
            try Task.checkCancellation()
            // V2: DiT hidden is concatenation
            let ditH1 = lmToDitProj(lmHidden)
            let ditH2 = resToDitProj(residualHidden)
            let ditH = MLX.concatenated([ditH1, ditH2], axis: -1) // (1, 2*H_dit)

            let condIn = prefixFeatCond.transposed(0, 2, 1) // (B, D, P)

            var predFeat = featDecoder.sample(
                mu: ditH,
                nTimesteps: inferenceTimesteps,
                patchSize: patchSize,
                cond: condIn,
                cfgValue: cfgValue
            )

            predFeat = predFeat.transposed(0, 2, 1) // (B, P, D)

            predFeatSeq.append(predFeat[0..., .newAxis, 0..., 0...]) // (B, 1, P, D)

            let currEmbed = featEncoder(predFeat[0..., .newAxis, 0..., 0...].asType(dtype))
            let currEmbedProj = encToLmProj(currEmbed)

            // Stop prediction
            let stopLogits = stopHead(silu(stopProj(lmHidden)))
            let stopFlag = MLX.argMax(stopLogits, axis: -1).item(Int.self)
            if i > minTokens && stopFlag == 1 {
                break
            }

            // Autoregressive step
            let newLmOut: MLXArray
            (newLmOut, lmCache) = baseLM(inputsEmbeds: currEmbedProj, cache: lmCache)

            lmHidden = newLmOut[0..., -1, 0...]
            lmHidden = fsqLayer(lmHidden)

            // V2: fusion_concat_proj for residual step
            let currResInput = fusionConcatProj(
                MLX.concatenated([lmHidden[0..., .newAxis, 0...], currEmbedProj], axis: -1)
            )
            let newResOut: MLXArray
            (newResOut, resCache) = residualLM(inputsEmbeds: currResInput, cache: resCache)
            residualHidden = newResOut[0..., -1, 0...]

            prefixFeatCond = predFeat

            eval(lmHidden, residualHidden, predFeat)
        }

        guard !predFeatSeq.isEmpty else {
            throw AudioGenerationError.invalidInput("Model generated no audio patches")
        }

        // Decode
        let allFeats = MLX.concatenated(predFeatSeq, axis: 1) // (B, Total, P, D)
        let B = allFeats.dim(0)
        let allFeatsFlat = allFeats.reshaped(B, -1, featDim) // (B, Total*P, D)

        var audio = audioVAE.decode(allFeatsFlat).flattened()

        // Trim the continuation prefix
        if hasContinuation {
            let trimSamples = patchSize * audioVAE.decodeChunkSize * (streamingPrefixLen - 1)
            if trimSamples < audio.dim(0) {
                audio = audio[trimSamples...]
            }
        }
        eval(audio)
        return audio
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MaterializedArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MaterializedArray {
        let audio = try generateAudio(
            text: text,
            refAudio: refAudio,
            cfgValue: config.cfgScale,
            inferenceTimesteps: config.inferenceTimesteps,
            maxTokens: generationParameters.maxTokens ?? config.maxLength
        )
        return audio.materialized()
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MaterializedArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let startTime = Date()
                let audio = try await self.generate(
                    text: text, voice: voice, refAudio: refAudio,
                    refText: refText, language: language,
                    generationParameters: generationParameters
                )
                let generateTime = Date().timeIntervalSince(startTime)

                let info = AudioGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: audio.dim(audio.ndim - 1),
                    prefillTime: 0,
                    generateTime: generateTime,
                    tokensPerSecond: Double(audio.dim(audio.ndim - 1)) / max(generateTime, 0.001),
                    peakMemoryUsage: 0
                )
                continuation.yield(.audio(audio))
                continuation.yield(.info(info))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        continuation.onTermination = { @Sendable _ in
            task.cancel()
        }

        return stream
    }

    // MARK: - Weight Sanitization

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Delegate VAE sanitization
        var vaeWeights = [String: MLXArray]()
        var otherWeights = [String: MLXArray]()

        for (key, value) in weights {
            if key.hasPrefix("audio_vae.") {
                let subKey = String(key.dropFirst("audio_vae.".count))
                vaeWeights[subKey] = value
            } else {
                otherWeights[key] = value
            }
        }

        // Sanitize VAE weights
        if !vaeWeights.isEmpty {
            let sanitizedVAE = audioVAE.sanitize(weights: vaeWeights)
            for (key, value) in sanitizedVAE {
                otherWeights["audio_vae.\(key)"] = value
            }
        }

        // Extract sr_boundaries buffer
        let srKey = "audio_vae.decoder.srBoundaries"
        if let srBounds = otherWeights.removeValue(forKey: srKey) {
            audioVAE.decoder.srBoundaries = srBounds
        }
        // Also handle the Python key name
        let srKey2 = "audio_vae.decoder._sr_boundaries"
        if let srBounds = otherWeights.removeValue(forKey: srKey2) {
            audioVAE.decoder.srBoundaries = srBounds
        }

        // Remap snake_case Python keys → camelCase Swift property names
        let keyMap = [
            "special_token": "specialToken",
        ]
        var result = [String: MLXArray]()
        for (key, value) in otherWeights {
            var mappedKey = key
            for (snake, camel) in keyMap {
                if mappedKey.hasSuffix(".\(snake)") {
                    mappedKey = String(mappedKey.dropLast(snake.count)) + camel
                } else if mappedKey == snake {
                    mappedKey = camel
                }
            }
            result[mappedKey] = value
        }

        // Add RoPE buffers if missing
        let flatParams = parameters().flattened()
        for (key, value) in flatParams {
            if result[key] == nil && key.contains("rope") {
                result[key] = value
            }
        }

        let modelParamKeys = Set(flatParams.map(\.0))
        let quantSuffixes = [".scales", ".biases"]
        let unmappedKeys = result.keys.filter { key in
            !modelParamKeys.contains(key) && !quantSuffixes.contains(where: { key.hasSuffix($0) })
        }
        if !unmappedKeys.isEmpty {
            print("[VoxCPM2] \(unmappedKeys.count) weight key(s) not mapped to model parameters: \(unmappedKeys.sorted().prefix(10))")
        }

        return result
    }

    // MARK: - Factory

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> VoxCPM2Model {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )

        return try await fromModelDirectory(modelDir, hfToken: hfToken)
    }

    public static func fromModelDirectory(
        _ modelDir: URL,
        hfToken: String?
    ) async throws -> VoxCPM2Model {
        // Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        let config: VoxCPM2Configuration
        if FileManager.default.fileExists(atPath: configURL.path) {
            let configData = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(VoxCPM2Configuration.self, from: configData)
        } else {
            config = VoxCPM2Configuration()
        }

        // Load tokenizer before model construction (tokenizer is let)
        let tokenizer: Tokenizers.Tokenizer?
        do {
            tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            print("[VoxCPM2] Warning: Could not load tokenizer: \(error)")
            tokenizer = nil
        }

        let model = VoxCPM2Model(config, tokenizer: tokenizer)

        // Load weights (single or sharded)
        let weights = try loadWeights(modelDir: modelDir)

        // Sanitize
        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantization (only base_lm and residual_lm)
        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard sanitizedWeights["\(path).scales"] != nil else { return nil }
                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path)
                {
                    return layerQuant.asTuple
                }
                return config.quantization?.asTuple
            }
        }

        // Update model parameters
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [])
        eval(model)

        return model
    }

    private static func loadWeights(modelDir: URL) throws -> [String: MLXArray] {
        let singleURL = modelDir.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: singleURL.path) {
            return try MLX.loadArrays(url: singleURL)
        }

        // Sharded weights
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !safetensorFiles.isEmpty else {
            throw AudioGenerationError.modelNotInitialized(
                "No .safetensors files found in \(modelDir.path)"
            )
        }

        var allWeights = [String: MLXArray]()
        for file in safetensorFiles {
            let shard = try MLX.loadArrays(url: file)
            for (key, value) in shard {
                allWeights[key] = value
            }
        }
        return allWeights
    }
}
