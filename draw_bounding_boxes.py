from PIL import ImageDraw
from torchvision.transforms import ToPILImage

def draw_bounding_boxes(
    image_tensor,
    translation_params,
    output_path,
    patch_size,
    resize_dim=(512, 512),
    bbox_color="red",
    line_thickness=2
):
    image = ToPILImage()(image_tensor)
    image = image.resize(resize_dim)

    orig_w, orig_h = image_tensor.shape[2], image_tensor.shape[1]
    scale_x = resize_dim[0] / orig_w
    scale_y = resize_dim[1] / orig_h

    draw = ImageDraw.Draw(image)

    for i in range(translation_params.size(1)):
        center_x = (translation_params[i, 0].item() + 1) / 2 * orig_w * scale_x
        center_y = (translation_params[i, 1].item() + 1) / 2 * orig_h * scale_y

        left = center_x - patch_size / 2 * scale_x
        top = center_y - patch_size / 2 * scale_y
        right = center_x + patch_size / 2 * scale_x
        bottom = center_y + patch_size / 2 * scale_y

        draw.rectangle(
            [left, top, right, bottom],
            outline=bbox_color,
            width=line_thickness
        )

    image.save(output_path)
