def frame_error_rate(failures: int, total_frames: int) -> float:
    if total_frames == 0:
        return 0.0
    return failures / total_frames