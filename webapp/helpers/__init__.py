"""
WebApp Helper Modules Package
Contains extracted functionality from app.py for better modularity
"""

from .video_processing import (
    process_video_segment,
    process_temporal_video_segment,
    calculate_video_chunks,
    get_video_caption_data
)

from .streaming_helpers import (
    process_video_for_caption_streaming,
    process_video_for_caption_optimized,
    process_video_for_caption,
    combine_captions_to_paragraph,
    format_captions_as_list
)

from .temporal_analysis import (
    convert_frames_to_timestamps,
    extract_temporal_events,
    format_temporal_analysis,
    estimate_timestamp_from_text,
    extract_sequential_events
)

from .utils import (
    serve_original_video,
    serve_video,
    get_available_videos,
    validate_video_file,
    format_file_size,
    format_duration,
    create_response,
    sanitize_filename
)

from .gpt_timeline_processor import (
    GPTTimelineProcessor
)

__all__ = [
    # Video processing
    'process_video_segment',
    'process_temporal_video_segment',
    'calculate_video_chunks',
    'get_video_caption_data',

    # Streaming helpers
    'process_video_for_caption_streaming',
    'process_video_for_caption_optimized',
    'process_video_for_caption',
    'combine_captions_to_paragraph',
    'format_captions_as_list',

    # Temporal analysis
    'convert_frames_to_timestamps',
    'extract_temporal_events',
    'format_temporal_analysis',
    'estimate_timestamp_from_text',
    'extract_sequential_events',

    # GPT Timeline processing
    'GPTTimelineProcessor',

    # Utilities
    'serve_original_video',
    'serve_video',
    'get_available_videos',
    'validate_video_file',
    'format_file_size',
    'format_duration',
    'create_response',
    'sanitize_filename'
]