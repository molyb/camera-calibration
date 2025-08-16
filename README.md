# Camera calibration

## Requirements

- [uv](https://docs.astral.sh/uv/)

## キャリブレーションボードの生成
```bash
uv run generate_calibration_board.py
```

`calibration_board.pdf` が生成されるので、原寸大で印刷して使用する。
ただし、現状印刷時にわずかに寸法がズレが生じているているため修正が必要。

## キャリブレーション

キャリブレーションボードを撮影した画像を格納したディレクトリを引数として `calibrate.py` を実行。
`./image` ディレクトリに画像を格納した場合のコマンドは以下。

```bash
uv run calibrate.py ./image
```

出力例
```
$ uv run .\calibrate.py image
2025-08-17 15:45:01,994 [INFO] root: Found 126 images for calibration.
2025-08-17 15:45:10,044 [INFO] root: Starting calibration...
2025-08-17 15:46:02,709 [INFO] root: Pruning iteration 1: dropping 10 views (error > 1.9377)
2025-08-17 15:46:56,411 [INFO] root: Pruning iteration 2: dropping 4 views (error > 1.8505)
2025-08-17 15:47:39,481 [INFO] root: Calibration successful!
2025-08-17 15:47:39,481 [INFO] root: Image size: (1920, 1080)
2025-08-17 15:47:39,481 [INFO] root: Camera Matrix:
[[2.59680558e+03 0.00000000e+00 8.86967450e+02]
 [0.00000000e+00 2.60159080e+03 5.24608058e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
2025-08-17 15:47:39,481 [INFO] root: Distortion Coefficients: [[ 0.13853824  0.37345717 -0.00365482 -0.00524672 -2.3823727 ]]
2025-08-17 15:47:39,489 [INFO] root: Calibration results saved to 'calibration_results.npz'
```
