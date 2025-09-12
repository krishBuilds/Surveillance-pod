"""
Temporal Analysis Helpers
Extracted from app.py to improve modularity
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def convert_frames_to_timestamps(frame_response: str, start_time: float, end_time: float, 
                                 num_frames: int, fps_sampling: float) -> str:
    """
    Convert frame-based analysis to timestamp-based analysis
    
    Args:
        frame_response: Original response with frame references
        start_time: Start time in seconds
        end_time: End time in seconds
        num_frames: Number of frames processed
        fps_sampling: FPS sampling rate
        
    Returns:
        Response with timestamps instead of frame references
    """
    logger.info(f"Converting frame response to timestamps: {start_time:.1f}s-{end_time:.1f}s")
    
    try:
        # Calculate the actual duration covered
        duration = min(num_frames / fps_sampling, end_time - start_time)
        
        # Replace frame references with time references
        converted_response = frame_response
        
        # Pattern to match frame references like "Frame 1", "frame 5", etc.
        frame_patterns = [
            r'(Frame|frame)\s*(\d+)',
            r'(at frame|At frame)\s*(\d+)',
            r'(in frame|In frame)\s*(\d+)',
            r'(\d+)(st|nd|rd|th)\s*frame',
        ]
        
        for pattern in frame_patterns:
            matches = re.finditer(pattern, frame_response, re.IGNORECASE)
            
            for match in matches:
                frame_num = int(match.group(2))
                
                # Calculate timestamp for this frame
                if frame_num <= num_frames:
                    timestamp = start_time + (frame_num - 1) / fps_sampling
                    timestamp = min(timestamp, end_time)
                    
                    # Replace with timestamp
                    time_ref = f"at {timestamp:.1f}s"
                    converted_response = converted_response.replace(match.group(0), time_ref)
        
        # Replace other frame-related terms
        frame_terms = [
            (r'(beginning frames|early frames)', f"beginning ({start_time:.1f}s-{start_time + 5:.1f}s)"),
            (r'(middle frames|mid frames)', f"middle section ({start_time + duration/3:.1f}s-{start_time + 2*duration/3:.1f}s)"),
            (r'(ending frames|later frames|final frames)', f"ending ({start_time + 2*duration/3:.1f}s-{end_time:.1f}s)"),
            (r'(throughout the frames|across frames)', f"throughout this {duration:.1f}s segment"),
        ]
        
        for pattern, replacement in frame_terms:
            converted_response = re.sub(pattern, replacement, converted_response, flags=re.IGNORECASE)
        
        # Add temporal context header
        temporal_context = f"[Temporal Analysis: {start_time:.1f}s - {end_time:.1f}s]\n\n"
        
        return temporal_context + converted_response
        
    except Exception as e:
        logger.error(f"Error converting frames to timestamps: {str(e)}", exc_info=True)
        # Return original response if conversion fails
        return f"[Temporal Context: {start_time:.1f}s - {end_time:.1f}s]\n\n{frame_response}"

def extract_temporal_events(response_text: str, start_time: float, end_time: float) -> List[str]:
    """
    Extract temporal events and milestones from AI response
    
    Args:
        response_text: AI-generated response text
        start_time: Start time of the segment
        end_time: End time of the segment
        
    Returns:
        List of extracted events with timing information
    """
    logger.info(f"Extracting temporal events from {start_time:.1f}s-{end_time:.1f}s segment")
    
    events = []
    
    try:
        # Common temporal indicators
        temporal_patterns = [
            r'(?:at|around|approximately|about)\s+(\d+(?:\.\d+)?)\s*s(?:econds?)?',
            r'(?:in the|at the)\s+(beginning|start|middle|end)\s*(?:of\s+the\s+segment)?',
            r'(?:first|initially|starts?\s+with|begins?\s+with)',
            r'(?:then|next|after\s+(?:that|this)|subsequently)',
            r'(?:later|finally|eventually|ends?\s+with|concludes?\s+with)',
            r'(?:during|throughout)\s+(?:the\s+)?(?:first|second|third|last)\s+(?:half|part|quarter)',
            r'(\d+(?:\.\d+)?)\s*s(?:econds?)?\s+(?:into|through|during)',
        ]
        
        segment_duration = end_time - start_time
        
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, response_text, re.IGNORECASE)
            
            for match in matches:
                match_group = match.group(0)
                
                # Calculate approximate timestamp
                timestamp = estimate_timestamp_from_text(match_group, start_time, segment_duration)
                
                if start_time <= timestamp <= end_time:
                    events.append({
                        'timestamp': timestamp,
                        'description': match_group.strip(),
                        'confidence': 'medium'
                    })
        
        # Look for sequential indicators (first, then, finally, etc.)
        sequential_events = extract_sequential_events(response_text, start_time, end_time)
        events.extend(sequential_events)
        
        # Sort by timestamp and remove duplicates
        unique_events = []
        seen_timestamps = set()
        
        for event in sorted(events, key=lambda x: x['timestamp']):
            rounded_time = round(event['timestamp'], 1)
            if rounded_time not in seen_timestamps:
                seen_timestamps.add(rounded_time)
                unique_events.append(event)
        
        logger.info(f"Extracted {len(unique_events)} temporal events")
        return unique_events
        
    except Exception as e:
        logger.error(f"Error extracting temporal events: {str(e)}", exc_info=True)
        return []

def estimate_timestamp_from_text(text: str, start_time: float, duration: float) -> float:
    """
    Estimate timestamp from temporal text description
    
    Args:
        text: Text containing temporal reference
        start_time: Segment start time
        duration: Segment duration
        
    Returns:
        Estimated timestamp
    """
    text_lower = text.lower()
    
    # Exact timestamp matches
    time_match = re.search(r'(\d+(?:\.\d+)?)\s*s(?:econds?)?', text_lower)
    if time_match:
        exact_time = float(time_match.group(1))
        return start_time + exact_time
    
    # Relative position matches
    if any(phrase in text_lower for phrase in ['beginning', 'start', 'first', 'initially']):
        return start_time + duration * 0.1
    elif any(phrase in text_lower for phrase in ['middle', 'mid']):
        return start_time + duration * 0.5
    elif any(phrase in text_lower for phrase in ['end', 'final', 'last', 'concludes']):
        return start_time + duration * 0.9
    
    # Sequential markers
    if any(phrase in text_lower for phrase in ['then', 'next', 'after']):
        return start_time + duration * 0.6
    
    # Default to middle of segment
    return start_time + duration * 0.5

def extract_sequential_events(response_text: str, start_time: float, end_time: float) -> List[Dict]:
    """
    Extract events based on sequential markers (first, then, finally)
    
    Args:
        response_text: AI response text
        start_time: Segment start time
        end_time: Segment end time
        
    Returns:
        List of sequentially ordered events
    """
    events = []
    duration = end_time - start_time
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', response_text)
    
    sequential_markers = {
        'beginning': 0.1,
        'start': 0.1,
        'first': 0.2,
        'initially': 0.1,
        'then': 0.4,
        'next': 0.4,
        'after': 0.5,
        'subsequently': 0.6,
        'later': 0.7,
        'finally': 0.9,
        'end': 0.9,
        'last': 0.9,
        'concludes': 0.9,
    }
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Find sequential markers
        for marker, position in sequential_markers.items():
            if marker in sentence.lower():
                timestamp = start_time + (duration * position)
                
                # Extract the event description (clean up the sentence)
                event_desc = re.sub(r'^(?:Then|Next|After|Finally|Initially|First|Last)\s*,?\s*', '', sentence, flags=re.IGNORECASE)
                event_desc = event_desc.strip()
                
                if event_desc:
                    events.append({
                        'timestamp': timestamp,
                        'description': event_desc,
                        'confidence': 'high' if marker in ['first', 'then', 'finally'] else 'medium',
                        'sequential_order': i
                    })
                break
    
    return events

def format_temporal_analysis(events: List[Dict], start_time: float, end_time: float) -> str:
    """
    Format extracted temporal events into a readable timeline
    
    Args:
        events: List of temporal events
        start_time: Segment start time
        end_time: Segment end time
        
    Returns:
        Formatted temporal timeline
    """
    if not events:
        return "No distinct temporal events identified."
    
    sorted_events = sorted(events, key=lambda x: x['timestamp'])
    
    timeline_lines = [f"Timeline ({start_time:.1f}s - {end_time:.1f}s):"]
    
    for event in sorted_events:
        timestamp = event['timestamp']
        description = event['description']
        confidence = event.get('confidence', 'medium')
        
        confidence_icon = "●" if confidence == 'high' else "○"
        timeline_lines.append(f"  {confidence_icon} {timestamp:.1f}s: {description}")
    
    return "\n".join(timeline_lines)