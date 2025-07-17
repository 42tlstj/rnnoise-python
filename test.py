from rnnoise.rnnoise import RNNoise
import numpy as np
import soundfile as sf

def main():
    denoiser = RNNoise()

    signal, sr = sf.read("녹음 (4).wav")
    assert sr == 48000, "샘플레이트는 48000이어야 합니다."

    # 스테레오 → 모노
    if signal.ndim > 1:
        signal = signal[:, 0]

    output_frames = []
    frame_length = 480

    for i in range(0, len(signal), frame_length):
        frame = signal[i:i+frame_length]
        # process_frame 에서 자동 패딩 처리됨
        denoised = denoiser.process_frame(frame)
        output_frames.append(denoised)

    output = np.concatenate(output_frames)
    sf.write("denoised13.wav", output, sr)
    print("denoised.wav 파일 생성 완료!")

if __name__ == "__main__":
    main()