AUDIO_WORKLOAD_HOURS = [2.5, 37.5, 375.0, 56000.0]
OCR_WORKLOAD_IMAGES = [(50, 80), (750, 1200), (7500, 12000), (1120000, 1790000)]


def audio_projection(audio_seconds: float, inference_seconds: float) -> dict:
    rate = audio_seconds / inference_seconds
    return {
        "real_time_factor": inference_seconds / audio_seconds,
        "audio_hours_per_wall_hour": rate,
        "projected_wall_hours": {str(hours): hours / rate for hours in AUDIO_WORKLOAD_HOURS},
    }

def ocr_projection(image_count: int, inference_seconds: float) -> dict:
    rate = image_count / inference_seconds * 3600
    return {
        "seconds_per_image": inference_seconds / image_count,
        "images_per_hour": rate,
        "projected_wall_hours": {
            f"{low}-{high}": [low / rate, high / rate] for low, high in OCR_WORKLOAD_IMAGES
        },
    }
