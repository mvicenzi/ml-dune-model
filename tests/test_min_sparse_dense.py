import torch

try:
    from warpconvnet.geometry.types.voxels import Voxels
except ModuleNotFoundError as exc:
    raise SystemExit(
        "warpconvnet is not installed in this environment. "
        "Install/activate it, then rerun this script."
    ) from exc


def main() -> None:
    # Dense input: [B, C, H, W]
    dense = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 5.0],
                    [0.0, 0.0, 6.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    # Dense -> sparse (WarpConvNet)
    vox = Voxels.from_dense(dense)

    print("Dense side")
    print(f"  tensor shape:        {tuple(dense.shape)}")

    print("\nSparse side (Voxels)")
    print(f"  coordinate shape:    {tuple(vox.coordinate_tensor.shape)}")
    print(f"  feature shape:       {tuple(vox.feature_tensor.shape)}")
    print(f"  offsets shape:       {tuple(vox.offsets.shape)}")

    # Sparse -> dense
    dense_back = vox.to_dense(channel_dim=1, spatial_shape=(dense.shape[2], dense.shape[3]))

    print("\nBack to dense")
    print(f"  reconstructed shape: {tuple(dense_back.shape)}")

    max_abs_diff = (dense - dense_back).abs().max().item()
    print(f"  max abs diff:        {max_abs_diff:.6f}")


if __name__ == "__main__":
    main()
