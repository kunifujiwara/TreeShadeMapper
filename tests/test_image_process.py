import numpy as np

from tree_shade_mapper.image_process import image_to_2d_array, rgb_to_int


def test_rgb_to_int_preserves_full_rgb_value() -> None:
    assert rgb_to_int((70, 130, 180)) == 4620980
    assert rgb_to_int((107, 142, 35)) == 7048739


def test_image_to_2d_array_preserves_segmentation_color_ids(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "segmented.png"
    image = Image.fromarray(
        np.array([[(70, 130, 180), (107, 142, 35)]], dtype=np.uint8),
        mode="RGB",
    )
    image.save(image_path)

    array = image_to_2d_array(image_path)

    assert array[0, 0] == 4620980
    assert array[0, 1] == 7048739
