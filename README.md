使用：python demo_viser.py --config configs/attn_vis.yaml --image_folder examples/kitchen/images/


config关键点：
attn_heatmaps：开了就存类似于fastvggt那样的single token -> tokens in other images这样的attn map.
patchmap： 开了就存faster vggt那样full attention map (number of patch token * number of patch token)


queries: "all", patchmap模式下，一定要用“all”，但建议缩小数据集，容易爆显存，懒得修了。
queries: ""cam" | "cam+reg" | "indices", attn_heatmaps模式下用的，indices模式下，你给的对应token会在所属frame上hightlight出来