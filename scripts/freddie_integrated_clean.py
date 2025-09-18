#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Freddie 2.0 — Integrated Runtime
================================

Expressive robotic face control with real-time audio/vision and LLM wiring.

Subsystems
----------
• Mask control (serial): 16-ch PCA9685 servo rig (brows, eyes, lids, jaw, mouth, neck)
• Micro-movement engine: blinks, saccades, emotion micro-adjustments
• Neck tracking: MediaPipe face detection → PID to yaw/pitch servos
• Speech Emotion Recognition (SER): 3 s windows, mel/MFCC/pitch features, scikit-learn model
• Lip sync: phoneme→viseme mapping with energy-scaled jaw/mouth offsets
• STT: Whisper (faster-whisper) for commands, wake/sleep flow
• TTS: Piper CLI synthesis with interruptible playback
• Web tools: simple search/weather/time helpers for LLM calls
• Conversation memory: small rolling context persisted to disk

Usage (PowerShell)
------------------
# Continuous mode (listen/think/speak loop)
PS> python .\freddie_integrated_clean.py --continuous

# One-shot prompt (speak a reply once)
PS> python .\freddie_integrated_clean.py --prompt "Tell me a joke"

# Device/port overrides (examples)
PS> python .\freddie_integrated_clean.py --serial COM4 --cam 0

Environment
-----------
Python 3.10+ recommended on Windows.
Requires audio I/O (sounddevice/pyaudio), camera (OpenCV), and serial (pyserial).

Hardware Notes
--------------
ESP32-S3 + PCA9685 at 50 Hz. Jaw/eyes/lids/mouth mapped to channels 0–12; neck on 13–15.
Firmware accepts ASCII commands: PING, VERSION, NEUTRAL, POSE, RAW, JAW, POSES, SAVE, SETBASE, PRINTPOS.
"""

from __future__ import annotations
import os
import sys
import time
import json
import subprocess
import tempfile
import wave
import argparse
import threading
import signal
import pyaudio
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import joblib
import serial
import cv2
import mediapipe as mp
from google import genai

__author__  = "Rambod Taherian"
__email__   = "rambodt@uw.com"
__version__ = "1.0.0"
__license__ = "MIT"
__url__     = "https://github.com/rambodt/Freddie-2.0"

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phoneme_viseme_detector import PhonemeVisemeDetector
from web_search_functions import WebSearchFunctions

# Platform-specific imports
if sys.platform == 'win32':
    import msvcrt
    import ctypes

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Text-to-Speech Configuration
PIPER_EXE = r"C:\path\to\piper.exe"
VOICE_ONNX = r"C:\path\to\en_US-norman-medium.onnx"

# Speech Recognition Configuration
WHISPER_MODEL = "tiny.en"
SAMPLE_RATE = 16000
RECORD_MAX_S = 15
ALIGN_MS = 60

# Voice Activity Detection Configuration
VAD_BLOCK_MS = 30             # Audio block size in milliseconds
VAD_START_THRESH = 0.07       # RMS threshold to start recording
VAD_STOP_THRESH = 0.02        # RMS threshold for silence detection
VAD_STOP_DURATION_MS = 850    # Milliseconds of silence before stopping
NO_SPEECH_TIMEOUT = 3.0       # Seconds before timeout with no speech
MIN_BUFFER_S = 1.0            # Minimum buffer for SER analysis

# Speech Emotion Recognition Configuration
SER_MODEL = r# "ser model (.joblib) address" 
SER_LABELS = r# "ser model labels (.jason) address" 
SER_BIAS = {
    "neutral": -0.20,
    "happy": +0.55,
    "angry": +0.55,
    "sad": -0.25,
    "shocked": 2.0,
    "fear": -0.2,
    "disgust": +0.1,
}
SER_MIN_CONF = 0.30           # Minimum confidence for emotion detection
SER_EMA_ALPHA = 0.15          # Exponential moving average alpha (0-1)
SER_RMS_GATE = 0.07           # Minimum RMS for emotion processing
SER_WINDOW_S = 3.0            # Emotion analysis window in seconds
SER_UPDATE_INTERVAL = 0.25    # Seconds between emotion updates

# Wake Detection Configuration
WAKE_BLOCK_MS = 50            # Audio block size for wake detection
WAKE_MIN_ENERGY = 0.008       # Minimum RMS to process for wake words
WAKE_PROCESS_INTERVAL = 1.5   # Seconds between wake phrase checks
WAKE_BUFFER_SECONDS = 1.5     # Audio buffer size for wake detection

# Wake Phrases - Everything that should wake Freddie
WAKE_PHRASES = [
    "hey freddie",
    "hey freddy",
    "wake up freddie",
    "wake up freddy",
    "freddie wake up",
    "freddy wake up",
    "wake up",
    "freddie",
    "freddy"
]

# Sleep Commands
SLEEP_COMMANDS = [
    "go to sleep",
    "goto sleep", 
    "go sleep",
    "sleep mode",
    "sleep now",
    "take a nap",
    "goodnight",
    "good night"
]

# Sleep Mode Configuration
SLEEP_TIMEOUT = 30  # Seconds of inactivity before sleep

# Push-to-Talk Configuration
PTT_MARGIN_THRESH = 0.10     # Margin threshold for emotion confidence
PTT_SHOW_TOPK = 4            # Number of top emotions to show

# Serial Communication Configuration
SERIAL_PORT = "" # Your Arduino port "COM2" for example
SERIAL_BAUD = 2000000

# Servo Channel Mapping
CH_L_BROW_V = 0   # Left eyebrow vertical
CH_L_BROW_A = 1   # Left eyebrow angle
CH_R_BROW_V = 2   # Right eyebrow vertical
CH_R_BROW_A = 3   # Right eyebrow angle
CH_R_X = 4        # Right eye X position
CH_R_Y = 5        # Right eye Y position
CH_R_LID = 6      # Right eyelid
CH_L_X = 7        # Left eye X position
CH_L_Y = 8        # Left eye Y position
CH_L_LID = 9      # Left eyelid
CH_JAW = 10       # Jaw position
CH_R_MOUTH = 11   # Right mouth corner
CH_L_MOUTH = 12   # Left mouth corner

# Neck servo channels
CH_NECK_YAW = 13
CH_NECK_R = 14
CH_NECK_L = 15

# Viseme to Servo Offset Mapping
VISEME_OFFSETS = {
    "AI": {CH_JAW: 300, CH_R_MOUTH: 100, CH_L_MOUTH: -100},
    "E": {CH_JAW: 100, CH_R_MOUTH: -100, CH_L_MOUTH: 100},
    "O": {CH_JAW: 200, CH_R_MOUTH: 150, CH_L_MOUTH: -150},
    "U": {CH_JAW: 120, CH_R_MOUTH: 100, CH_L_MOUTH: -100},
    "MBP": {CH_JAW: -300, CH_R_MOUTH: 0, CH_L_MOUTH: 0},
    "FV": {CH_JAW: -300, CH_R_MOUTH: -60, CH_L_MOUTH: 60},
    "L": {CH_JAW: 120, CH_R_MOUTH: -80, CH_L_MOUTH: 80},
    "SZ": {CH_JAW: -300, CH_R_MOUTH: -50, CH_L_MOUTH: 50},
    "REST": {CH_JAW: 0, CH_R_MOUTH: 0, CH_L_MOUTH: 0},
}

# Global shutdown flag
shutdown_requested = False

# ============================================================================
# SHUTDOWN AND CLEANUP HANDLERS
# ============================================================================

def force_kill_process():
    """Force terminate the process immediately."""
    if sys.platform == 'win32':
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(1, False, os.getpid())
        kernel32.TerminateProcess(handle, 0)
    else:
        os.kill(os.getpid(), signal.SIGKILL)
    os._exit(0)


def cleanup_audio():
    """Clean up all audio streams before shutdown."""
    try:
        sd.abort()
        sd.stop()
        sd.default.reset()
        if hasattr(sd, '_last_callback'):
            sd._last_callback = None
    except:
        pass
    
    try:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            try:
                p.terminate()
            except:
                pass
    except:
        pass


def signal_handler(sig, frame):
    """Handle interrupt signals for graceful shutdown."""
    global shutdown_requested
    shutdown_requested = True
    
    print('\n\n╔══════════════════════════════════════════╗')
    print('    INTERRUPT - Force killing in 0.5 sec')
    print('╚══════════════════════════════════════════╝')
    
    def force_kill_timer():
        time.sleep(0.5)
        print("\n[FORCE KILL] Terminating process NOW!")
        force_kill_process()
    
    kill_thread = threading.Thread(target=force_kill_timer, daemon=True)
    kill_thread.start()
    
    try:
        cleanup_audio()
        cv2.destroyAllWindows()
    except:
        pass
    
    os._exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Windows-specific control handler
if sys.platform == 'win32':
    def win_handler(dwCtrlType):
        """Handle Windows control events."""
        if dwCtrlType in [0, 1, 2, 5, 6]:
            global shutdown_requested
            shutdown_requested = True
            print("\n[WIN32] Force terminating...")
            force_kill_process()
            return True
        return False
    
    HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint32)
    handler_func = HandlerRoutine(win_handler)
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCtrlHandler(handler_func, True)


def should_exit():
    """Check if shutdown was requested."""
    global shutdown_requested
    return shutdown_requested


# ============================================================================
# AUDIO PLAYBACK MANAGER
# ============================================================================

class InterruptibleAudioPlayer:
    """Manages audio playback with interrupt capability via spacebar."""
    
    def __init__(self):
        self.is_playing = False
        self.should_stop = False
        self.p = None
        self.stream = None
        self.play_thread = None
    
    def play_wav_interruptible(self, wav_path: str) -> bool:
        """
        Play WAV file with ability to interrupt by pressing spacebar.
        
        Args:
            wav_path: Path to WAV file
            
        Returns:
            True if completed normally, False if interrupted
        """
        self.should_stop = False
        self.is_playing = True
        
        self.play_thread = threading.Thread(
            target=self._play_audio_thread,
            args=(wav_path,),
            daemon=True
        )
        self.play_thread.start()
        
        # Monitor for spacebar press while playing
        while self.is_playing:
            if sys.platform == 'win32':
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ':  # Spacebar pressed
                        print("\n[INTERRUPTED] Stopping speech...")
                        self.stop()
                        while msvcrt.kbhit():
                            msvcrt.getch()
                        return False
            time.sleep(0.01)
        
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1)
        
        return not self.should_stop
    
    def _play_audio_thread(self, wav_path: str):
        """Thread function for audio playback."""
        wf = None
        try:
            wf = wave.open(wav_path, 'rb')
            self.p = pyaudio.PyAudio()
            
            self.stream = self.p.open(
                format=self.p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            
            while data and not self.should_stop:
                if not self.should_stop:
                    self.stream.write(data)
                data = wf.readframes(chunk_size)
            
        except Exception as e:
            if not self.should_stop:
                print(f"[Audio Error] {e}")
        finally:
            self.is_playing = False
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
            if self.p:
                try:
                    self.p.terminate()
                except:
                    pass
            if wf:
                try:
                    wf.close()
                except:
                    pass
    
    def stop(self):
        """Stop audio playback."""
        self.should_stop = True
        self.is_playing = False


# ============================================================================
# CONVERSATION MEMORY MANAGER
# ============================================================================

class ConversationMemory:
    """Manages conversation context and history with persistence."""
    
    def __init__(self, max_messages: int = 10, save_file: str = "freddie_conversation_memory.json"):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of message pairs to keep
            save_file: Path to save conversation history
        """
        self.max_messages = max_messages
        self.save_file = save_file
        self.messages = deque(maxlen=max_messages * 2)
        self.session_start = datetime.now()
        self.load_memory()
    
    def add_message(self, role: str, content: str, emotion: str = None):
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion
        }
        self.messages.append(message)
        self.save_memory()
    
    def get_context_for_llm(self) -> str:
        """Get formatted conversation history for the LLM."""
        if not self.messages:
            return ""
        
        context_lines = []
        for msg in self.messages:
            role = "User" if msg["role"] == "user" else "Freddie"
            emotion = f" (feeling {msg.get('emotion', 'neutral')})" if msg.get('emotion') and msg["role"] == "user" else ""
            context_lines.append(f"{role}{emotion}: {msg['content']}")
        
        return "\n".join(context_lines)
    
    def save_memory(self):
        """Save conversation memory to file."""
        try:
            data = {
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "messages": list(self.messages)
            }
            with open(self.save_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Memory] Failed to save: {e}")
    
    def load_memory(self):
        """Load conversation memory from file."""
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                saved_date = datetime.fromisoformat(data["session_start"]).date()
                if saved_date == datetime.now().date():
                    self.messages = deque(data["messages"], maxlen=self.max_messages * 2)
                    print(f"[Memory] Loaded {len(self.messages)} previous messages")
                else:
                    print("[Memory] Starting fresh conversation (new day)")
            except Exception as e:
                print(f"[Memory] Failed to load: {e}")
    
    def clear(self):
        """Clear conversation memory."""
        self.messages.clear()
        self.session_start = datetime.now()
        if os.path.exists(self.save_file):
            try:
                os.remove(self.save_file)
            except:
                pass


# ============================================================================
# SERIAL COMMUNICATION MANAGER
# ============================================================================

class SharedSerialManager:
    """Thread-safe serial communication manager for servo control."""
    
    def __init__(self, port: str = SERIAL_PORT, baud: int = SERIAL_BAUD):
        """
        Initialize serial connection.
        
        Args:
            port: Serial port name
            baud: Baud rate
        """
        self.port = port
        self.baud = baud
        self.ser = None
        self.lock = threading.Lock()
        self.connect()
    
    def connect(self):
        """Establish serial connection and perform handshake."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
            self.ser.dtr = False
            self.ser.rts = False
            time.sleep(0.5)
            print(f"[Serial] Connected to {self.port}")
            
            # Handshake
            self.ser.write(b"PING\n")
            time.sleep(0.1)
            response = self.ser.read(1000).decode('ascii', errors='ignore')
            if "PONG" in response:
                print("[Serial] Handshake OK")
        except Exception as e:
            print(f"[Serial] Failed to connect: {e}")
            raise
    
    def send(self, cmd: str):
        """
        Send command to serial port (thread-safe).
        
        Args:
            cmd: Command string to send
        """
        with self.lock:
            if self.ser and self.ser.is_open:
                self.ser.write((cmd.strip() + "\n").encode('ascii'))
    
    def close(self):
        """Close serial connection."""
        with self.lock:
            if self.ser:
                self.ser.write(b"NEUTRAL\n")
                time.sleep(0.2)
                self.ser.close()


# ============================================================================
# MICRO MOVEMENT CONTROLLER
# ============================================================================

class MicroMovementController:
    """Controls subtle micro-movements for lifelike appearance."""
    
    def __init__(self, serial_manager: SharedSerialManager):
        """
        Initialize micro-movement controller.
        
        Args:
            serial_manager: Serial communication manager
        """
        self.ser_mgr = serial_manager
        self.running = False
        self.thread = None
        self.current_emotion = "neutral"
        self.emotion_intensity = 0.0
        
        # Neutral servo positions
        self.NEUTRAL = np.array([
            990, 1420, 2070, 1510, 1290, 1210, 1560, 1880,
            1620, 1670, 1260, 1310, 1225, 1400, 1450, 1450
        ], dtype=np.float32)
        
        self.emotional_base = self.NEUTRAL.copy()
        self.current_pose = self.NEUTRAL.copy()
        self.last_blink = time.time()
    
    def start(self):
        """Start micro-movement thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._movement_loop, daemon=True)
            self.thread.start()
            print("[Micro] Started micro-movements")
    
    def stop(self):
        """Stop micro-movement thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            print("[Micro] Stopped micro-movements")
    
    def set_emotion(self, emotion: str, intensity: float):
        """
        Update emotion for micro-movements.
        
        Args:
            emotion: Emotion name
            intensity: Emotion intensity (0-1)
        """
        self.current_emotion = emotion
        self.emotion_intensity = intensity
        
        # Calculate emotional base pose
        self.emotional_base = self.NEUTRAL.copy()
        offsets = self._get_emotion_offsets(emotion, intensity)
        for ch, offset in offsets.items():
            self.emotional_base[ch] = max(500, min(2400, self.NEUTRAL[ch] + offset))
        
        self.current_pose = self.emotional_base.copy()
    
    def _get_emotion_offsets(self, emotion: str, intensity: float) -> Dict[int, float]:
        """Calculate servo offsets for given emotion."""
        offsets = {}
        z = max(0.0, min(1.0, intensity))
        
        if emotion == "happy":
            offsets[CH_R_MOUTH] = -525 * z
            offsets[CH_L_MOUTH] = 500 * z
            offsets[CH_L_BROW_V] = -20 * z
            offsets[CH_R_BROW_V] = 20 * z
            offsets[CH_R_LID] = -20 * z
            offsets[CH_L_LID] = 20 * z
            offsets[CH_R_BROW_A] = -100 * z
            offsets[CH_L_BROW_A] = 100 * z
        elif emotion == "sad":
            offsets[CH_R_MOUTH] = 400 * z
            offsets[CH_L_MOUTH] = -400 * z
            offsets[CH_L_BROW_V] = 50 * z
            offsets[CH_R_BROW_V] = -50 * z
            offsets[CH_L_BROW_A] = 200 * z
            offsets[CH_R_BROW_A] = -200 * z
            offsets[CH_R_LID] = 200 * z
            offsets[CH_L_LID] = -200 * z
        elif emotion == "angry":
            offsets[CH_R_MOUTH] = 400 * z
            offsets[CH_L_MOUTH] = -400 * z
            offsets[CH_L_BROW_V] = 70 * z
            offsets[CH_R_BROW_V] = -70 * z
            offsets[CH_L_BROW_A] = -250 * z
            offsets[CH_R_BROW_A] = 250 * z
            offsets[CH_R_LID] = 200 * z
            offsets[CH_L_LID] = -200 * z
        elif emotion == "shocked":
            offsets[CH_R_LID] = -400 * z
            offsets[CH_L_LID] = 400 * z
            offsets[CH_L_BROW_V] = -80 * z
            offsets[CH_R_BROW_V] = 80 * z
        
        return offsets
    
    def _movement_loop(self):
        """Main micro-movement loop."""
        while self.running:
            try:
                now = time.time()
                
                # Blinking (every 3-6 seconds)
                if now - self.last_blink > np.random.uniform(3, 6):
                    self._blink()
                    self.last_blink = now
                
                # Subtle eye movements (5% chance per cycle)
                if np.random.random() < 0.05:
                    self._subtle_eye_movement()
                
                # Emotion-based micro adjustments
                if self.current_emotion != "neutral" and self.emotion_intensity > 0:
                    self._emotion_micro_adjust()
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[Micro] Error: {e}")
                time.sleep(1)
    
    def _blink(self):
        """Perform a blink while maintaining emotional state."""
        if not self.running:
            return
        
        current_r_lid = int(self.emotional_base[CH_R_LID])
        current_l_lid = int(self.emotional_base[CH_L_LID])
        
        # Close lids
        self.ser_mgr.send(f"POSE {CH_R_LID} 2100")
        self.ser_mgr.send(f"POSE {CH_L_LID} 1139")
        time.sleep(0.15)
        
        # Return to emotional base positions
        self.ser_mgr.send(f"POSE {CH_R_LID} {current_r_lid}")
        self.ser_mgr.send(f"POSE {CH_L_LID} {current_l_lid}")
    
    def _subtle_eye_movement(self):
        """Generate small random eye movements."""
        if not self.running:
            return
        
        x_offset = np.random.randint(-50, 50)
        y_offset = np.random.randint(-30, 30)
        
        # Apply offsets
        self.ser_mgr.send(f"POSE {CH_R_X} {int(self.emotional_base[CH_R_X] + x_offset)}")
        self.ser_mgr.send(f"POSE {CH_L_X} {int(self.emotional_base[CH_L_X] + x_offset)}")
        self.ser_mgr.send(f"POSE {CH_R_Y} {int(self.emotional_base[CH_R_Y] - y_offset)}")
        self.ser_mgr.send(f"POSE {CH_L_Y} {int(self.emotional_base[CH_L_Y] + y_offset)}")
        
        # Return to base after a moment
        time.sleep(0.5)
        if self.running:
            self.ser_mgr.send(f"POSE {CH_R_X} {int(self.emotional_base[CH_R_X])}")
            self.ser_mgr.send(f"POSE {CH_L_X} {int(self.emotional_base[CH_L_X])}")
            self.ser_mgr.send(f"POSE {CH_R_Y} {int(self.emotional_base[CH_R_Y])}")
            self.ser_mgr.send(f"POSE {CH_L_Y} {int(self.emotional_base[CH_L_Y])}")
    
    def _emotion_micro_adjust(self):
        """Apply subtle emotion-based adjustments."""
        if not self.running:
            return
        
        if self.current_emotion == "happy":
            # Slight brow lift variation
            offset = int(15 * self.emotion_intensity * np.sin(time.time() * 0.5))
            self.ser_mgr.send(f"POSE {CH_L_BROW_V} {int(self.emotional_base[CH_L_BROW_V] + offset)}")
            self.ser_mgr.send(f"POSE {CH_R_BROW_V} {int(self.emotional_base[CH_R_BROW_V] - offset)}")
        elif self.current_emotion == "sad":
            # Subtle tremor in brows
            offset = int(10 * self.emotion_intensity * np.sin(time.time() * 2))
            self.ser_mgr.send(f"POSE {CH_L_BROW_A} {int(self.emotional_base[CH_L_BROW_A] + offset)}")
            self.ser_mgr.send(f"POSE {CH_R_BROW_A} {int(self.emotional_base[CH_R_BROW_A] - offset)}")


# ============================================================================
# NECK TRACKING CONTROLLER
# ============================================================================

class IntegratedNeckTracker:
    """Controls neck servos for head tracking using computer vision."""
    
    def __init__(self, serial_manager: SharedSerialManager):
        """
        Initialize neck tracker.
        
        Args:
            serial_manager: Serial communication manager
        """
        self.ser_mgr = serial_manager
        self.running = False
        self.paused = False
        self.thread = None
        self.cap = None
        self.mp_face = None
        
        # Servo configuration
        self.IDX_YAW = CH_NECK_YAW
        self.IDX_R = CH_NECK_R
        self.IDX_L = CH_NECK_L
        
        self.LIM = {
            self.IDX_YAW: (900, 2300),
            self.IDX_R: (900, 2300),
            self.IDX_L: (900, 2300)
        }
        
        self.NEU = {
            self.IDX_YAW: 1450,
            self.IDX_R: 1450,
            self.IDX_L: 1400
        }
        
        # PID controller parameters
        self.Kp_y, self.Ki_y, self.Kd_y = 1.98, 9.32, 1.40
        self.Kp_p, self.Ki_p, self.Kd_p = 4.70, 8.39, 1.53
        
        self.SCALE_Y = 100
        self.SCALE_P = 100
        self.DEADBAND = 0.15
        self.ERR_EMA = 0.95
        self.MAX_STEP = 15
    
    def start(self):
        """Start neck tracking thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.thread.start()
            print("[Neck] Started tracking")
    
    def stop(self):
        """Stop neck tracking and return to neutral."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        for ch, pos in self.NEU.items():
            self.ser_mgr.send(f"POSE {ch} {pos}")
            time.sleep(0.05)
        print("[Neck] Stopped")
    
    def pause(self):
        """Pause tracking but keep resources alive."""
        self.paused = True
        for ch, pos in self.NEU.items():
            self.ser_mgr.send(f"POSE {ch} {pos}")
            time.sleep(0.05)
        print("[Neck] Paused")
    
    def resume(self):
        """Resume tracking."""
        self.paused = False
        print("[Neck] Resumed")
    
    def _tracking_loop(self):
        """Main tracking loop using MediaPipe face detection."""
        try:
            # Initialize MediaPipe
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            
            # Initialize camera
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Simple PID controller
            class SimplePID:
                def __init__(self, kp, ki, kd):
                    self.kp, self.ki, self.kd = kp, ki, kd
                    self.integral = 0
                    self.last_error = 0
                
                def update(self, error, dt):
                    self.integral += error * dt
                    self.integral = max(-3, min(3, self.integral))
                    derivative = (error - self.last_error) / dt if dt > 0 else 0
                    self.last_error = error
                    return self.kp * error + self.ki * self.integral + self.kd * derivative
            
            yaw_pid = SimplePID(self.Kp_y, self.Ki_y, self.Kd_y)
            pitch_pid = SimplePID(self.Kp_p, self.Ki_p, self.Kd_p)
            
            last_positions = dict(self.NEU)
            smoothed_error = {"y": 0.0, "p": 0.0}
            t0 = time.perf_counter()
            
            # Go to neutral
            for ch, pos in self.NEU.items():
                self.ser_mgr.send(f"POSE {ch} {pos}")
                time.sleep(0.05)
            
            window_open = True
            
            while self.running:
                if self.paused:
                    if window_open:
                        cv2.destroyAllWindows()
                        window_open = False
                    time.sleep(0.1)
                    continue
                
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_face.process(rgb)
                
                t1 = time.perf_counter()
                dt = max(0.001, t1 - t0)
                t0 = t1
                
                if results.detections and len(results.detections) > 0:
                    detection = results.detections[0]
                    bb = detection.location_data.relative_bounding_box
                    
                    face_cx = bb.xmin + 0.5 * bb.width
                    face_cy = bb.ymin + 0.5 * bb.height
                    
                    # Calculate errors
                    raw_yerr = (face_cx - 0.5) * 2.0
                    raw_perr = -(face_cy - 0.5) * 2.0
                    
                    # Apply deadband
                    if abs(raw_yerr) < self.DEADBAND:
                        raw_yerr = 0.0
                    if abs(raw_perr) < self.DEADBAND:
                        raw_perr = 0.0
                    
                    # Smooth errors
                    smoothed_error["y"] = self.ERR_EMA * raw_yerr + (1 - self.ERR_EMA) * smoothed_error["y"]
                    smoothed_error["p"] = self.ERR_EMA * raw_perr + (1 - self.ERR_EMA) * smoothed_error["p"]
                    
                    # PID control
                    yaw_control = yaw_pid.update(smoothed_error["y"], dt) * self.SCALE_Y * -1
                    pitch_control = pitch_pid.update(smoothed_error["p"], dt) * self.SCALE_P
                    
                    # Calculate target positions
                    yaw_us = self.NEU[self.IDX_YAW] + yaw_control
                    r_us = self.NEU[self.IDX_R] + pitch_control
                    l_us = self.NEU[self.IDX_L] - pitch_control
                    
                    # Helper functions
                    def clamp(v, lo, hi):
                        return max(lo, min(hi, v))
                    
                    def step_limit(prev, target):
                        delta = target - prev
                        if abs(delta) > self.MAX_STEP:
                            return prev + self.MAX_STEP * (1 if delta > 0 else -1)
                        return target
                    
                    # Apply limits
                    yaw_us = clamp(yaw_us, *self.LIM[self.IDX_YAW])
                    r_us = clamp(r_us, *self.LIM[self.IDX_R])
                    l_us = clamp(l_us, *self.LIM[self.IDX_L])
                    
                    # Apply step limiting
                    yaw_us = step_limit(last_positions[self.IDX_YAW], yaw_us)
                    r_us = step_limit(last_positions[self.IDX_R], r_us)
                    l_us = step_limit(last_positions[self.IDX_L], l_us)
                    
                    # Send commands
                    self.ser_mgr.send(f"POSE {self.IDX_YAW} {int(yaw_us)}")
                    self.ser_mgr.send(f"POSE {self.IDX_R} {int(r_us)}")
                    self.ser_mgr.send(f"POSE {self.IDX_L} {int(l_us)}")
                    
                    # Update last positions
                    last_positions[self.IDX_YAW] = yaw_us
                    last_positions[self.IDX_R] = r_us
                    last_positions[self.IDX_L] = l_us
                
                # Show window
                if not window_open:
                    window_open = True
                
                cv2.putText(frame, "Neck Tracking Active", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Freddie Neck", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            
            # Clean up
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"[Neck] Error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MASK CONTROLLER
# ============================================================================

class IntegratedMaskController:
    """Controls facial expressions and lip synchronization."""
    
    def __init__(self, serial_manager: SharedSerialManager, micro_controller: MicroMovementController):
        """
        Initialize mask controller.
        
        Args:
            serial_manager: Serial communication manager
            micro_controller: Micro-movement controller
        """
        self.ser_mgr = serial_manager
        self.micro = micro_controller
        self.NEUTRAL = [990, 1420, 2070, 1510, 1290, 1210, 1560, 1880,
                        1620, 1670, 1260, 1310, 1225]
        self.current_emotion = "neutral"
        self.phoneme_detector = PhonemeVisemeDetector()
    
    def set_emotion(self, emotion: str, intensity: float):
        """
        Set facial emotion expression.
        
        Args:
            emotion: Emotion name
            intensity: Emotion intensity (0-1)
        """
        offsets = self._get_emotion_offsets(emotion, intensity)
        
        for ch, offset in offsets.items():
            target = self.NEUTRAL[ch] + offset
            target = max(500, min(2400, target))
            self.ser_mgr.send(f"POSE {ch} {int(target)}")
        
        self.current_emotion = emotion
        self.micro.set_emotion(emotion, intensity)
    
    def _get_emotion_offsets(self, emotion: str, intensity: float) -> Dict[int, float]:
        """Calculate servo offsets for emotion expression."""
        offsets = {}
        z = max(0.0, min(1.0, intensity))
        
        if emotion == "happy":
            offsets[CH_R_MOUTH] = -525 * z
            offsets[CH_L_MOUTH] = 500 * z
            offsets[CH_L_BROW_V] = -60 * z
            offsets[CH_R_BROW_V] = 60 * z
            offsets[CH_R_LID] = -20 * z
            offsets[CH_L_LID] = 20 * z
            offsets[CH_R_BROW_A] = -100 * z
            offsets[CH_L_BROW_A] = 100 * z
        elif emotion == "sad":
            offsets[CH_R_MOUTH] = 400 * z
            offsets[CH_L_MOUTH] = -400 * z
            offsets[CH_L_BROW_V] = 50 * z
            offsets[CH_R_BROW_V] = -50 * z
            offsets[CH_L_BROW_A] = 200 * z
            offsets[CH_R_BROW_A] = -200 * z
            offsets[CH_R_LID] = 200 * z
            offsets[CH_L_LID] = -200 * z
        elif emotion == "angry":
            offsets[CH_R_MOUTH] = 400 * z
            offsets[CH_L_MOUTH] = -400 * z
            offsets[CH_L_BROW_V] = 70 * z
            offsets[CH_R_BROW_V] = -70 * z
            offsets[CH_L_BROW_A] = -250 * z
            offsets[CH_R_BROW_A] = 250 * z
            offsets[CH_R_LID] = 200 * z
            offsets[CH_L_LID] = -200 * z
        elif emotion == "shocked":
            offsets[CH_R_LID] = -400 * z
            offsets[CH_L_LID] = 400 * z
            offsets[CH_L_BROW_V] = -80 * z
            offsets[CH_R_BROW_V] = 80 * z
        
        return offsets
    
    def set_viseme_pose(self, viseme: str, intensity: float):
        """
        Set mouth shape for viseme.
        
        Args:
            viseme: Viseme name
            intensity: Viseme intensity (0-1)
        """
        if viseme in VISEME_OFFSETS:
            offsets = VISEME_OFFSETS[viseme]
            for ch, offset in offsets.items():
                target = self.NEUTRAL[ch] + int(offset * intensity)
                target = max(500, min(2400, target))
                self.ser_mgr.send(f"POSE {ch} {int(target)}")
    
    def set_jaw(self, us: int):
        """Direct jaw position control."""
        self.ser_mgr.send(f"JAW {us}")
    
    def sleep_position(self):
        """Move to sleep position."""
        self.ser_mgr.send(f"POSE {CH_R_LID} 2100")
        self.ser_mgr.send(f"POSE {CH_L_LID} 1139")
        self.ser_mgr.send(f"POSE {CH_JAW} 1200")
        self.ser_mgr.send(f"POSE {CH_L_BROW_V} {int(self.NEUTRAL[CH_L_BROW_V])}")
        self.ser_mgr.send(f"POSE {CH_R_BROW_V} {int(self.NEUTRAL[CH_R_BROW_V])}")
    
    def wake_animation(self):
        """Perform wake-up animation."""
        for i in range(5):
            lid_r = 1700 - (i * 100)
            lid_l = 1300 + (i * 100)
            self.ser_mgr.send(f"POSE {CH_R_LID} {lid_r}")
            self.ser_mgr.send(f"POSE {CH_L_LID} {lid_l}")
            time.sleep(0.1)
        
        self.set_emotion("happy", 0.3)
        time.sleep(0.5)
        self.neutral()
    
    def neutral(self):
        """Return to neutral position."""
        self.ser_mgr.send("NEUTRAL")
        self.current_emotion = "neutral"
        self.micro.set_emotion("neutral", 0.0)


# ============================================================================
# SLEEP MODE CONTROLLER
# ============================================================================

class SleepModeController:
    """
    Manages sleep mode based on inactivity with speech-aware timing.
    
    This controller prevents the system from sleeping while Freddie is speaking,
    ensuring conversations aren't interrupted by sleep mode.
    """
    
    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize sleep controller.
        
        Args:
            timeout_seconds: Seconds of inactivity before sleep
        """
        self.timeout = timeout_seconds
        self.last_activity = time.time()
        self.asleep = False
        self.lock = threading.Lock()
        self.force_sleep = False
        self.freddie_speaking = False  # Track when Freddie is speaking
        
    def activity(self):
        """Mark user activity and wake if asleep."""
        with self.lock:
            self.last_activity = time.time()
            was_asleep = self.asleep
            self.asleep = False
            self.force_sleep = False
            if was_asleep:
                print("[Sleep] Waking up from activity")
    
    def freddie_started_speaking(self):
        """
        Mark that Freddie started speaking.
        Pauses the inactivity timer to prevent sleep during speech.
        """
        with self.lock:
            self.freddie_speaking = True
            print("[Sleep] Timer paused - Freddie speaking")
    
    def freddie_stopped_speaking(self):
        """
        Mark that Freddie stopped speaking.
        Restarts the inactivity timer from the current time.
        """
        with self.lock:
            self.freddie_speaking = False
            self.last_activity = time.time()  # Reset timer from NOW
            print("[Sleep] Timer restarted - Freddie finished speaking")
    
    def check_sleep(self) -> bool:
        """
        Check if should go to sleep due to inactivity.
        
        Returns:
            True if sleep should be initiated, False otherwise
        """
        with self.lock:
            if self.force_sleep:
                return False  # Already sleeping
            
            if self.freddie_speaking:
                return False  # Don't sleep while Freddie is talking
            
            if not self.asleep:
                inactive_time = time.time() - self.last_activity
                if inactive_time > self.timeout:
                    print(f"[Sleep] Inactive for {inactive_time:.0f} seconds")
                    self.asleep = True
                    return True
            return False
    
    def is_asleep(self) -> bool:
        """Check if currently asleep."""
        with self.lock:
            return self.asleep or self.force_sleep
    
    def wake(self):
        """Force wake up."""
        with self.lock:
            was_asleep = self.asleep or self.force_sleep
            self.asleep = False
            self.force_sleep = False
            self.freddie_speaking = False  # Reset speaking flag on wake
            self.last_activity = time.time()
            if was_asleep:
                print("[Sleep] Waking up!")
            return was_asleep
    
    def go_to_sleep(self):
        """Force sleep mode."""
        with self.lock:
            self.asleep = True
            self.force_sleep = True
            self.freddie_speaking = False  # Reset speaking flag
            print("[Sleep] Going to sleep (explicit command)")
            return True
    
    def reset_timer(self):
        """Reset inactivity timer without waking."""
        with self.lock:
            if not self.freddie_speaking:  # Only reset if not speaking
                self.last_activity = time.time()


# ============================================================================
# EMOTION PERSISTENCE CONTROLLER
# ============================================================================

class EmotionPersistenceController:
    """Manages emotion persistence and natural transitions."""
    
    def __init__(self, mask_controller: IntegratedMaskController, micro_controller: MicroMovementController):
        """
        Initialize emotion persistence.
        
        Args:
            mask_controller: Mask controller instance
            micro_controller: Micro-movement controller instance
        """
        self.mask = mask_controller
        self.micro = micro_controller
        self.current_emotion = "neutral"
        self.current_intensity = 0.0
        self.target_emotion = "neutral"
        self.target_intensity = 0.0
        self.decay_thread = None
        self.running = False
        
        # Emotion-specific persistence times (seconds)
        self.persistence_times = {
            "happy": 3.5,
            "sad": 4.0,
            "angry": 2.5,
            "shocked": 2.0,
            "neutral": 1.0
        }
        
        # Decay rates for each emotion
        self.decay_rates = {
            "happy": 0.15,
            "sad": 0.12,
            "angry": 0.20,
            "shocked": 0.25,
            "neutral": 0.30
        }
    
    def set_emotion(self, emotion: str, intensity: float, persist: bool = True):
        """
        Set emotion with optional persistence.
        
        Args:
            emotion: Emotion name
            intensity: Emotion intensity (0-1)
            persist: Whether to enable persistence
        """
        if self.decay_thread and self.decay_thread.is_alive():
            if self.current_emotion == emotion:
                print(f"[Emotion] Already persisting {emotion}, skipping")
                return
            else:
                print(f"[Emotion] Switching from {self.current_emotion} to {emotion}")
                self.running = False
                self.decay_thread.join(timeout=0.5)
        
        self.target_emotion = emotion
        self.target_intensity = intensity
        
        self.mask.set_emotion(emotion, intensity)
        self.current_emotion = emotion
        self.current_intensity = intensity
        
        if persist:
            self._start_persistence()
    
    def _start_persistence(self):
        """Start the persistence and decay process."""
        self.running = True
        self.decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
        self.decay_thread.start()
    
    def _decay_loop(self):
        """Gradually decay emotion after persistence period."""
        import random
        
        # Hold the emotion for persistence time
        persistence_time = self.persistence_times.get(self.current_emotion, 2.0)
        persistence_time *= random.uniform(0.8, 1.2)  # Add variation
        
        print(f"[Emotion] Holding {self.current_emotion} for {persistence_time:.1f}s")
        time.sleep(persistence_time)
        
        if not self.running:
            return
        
        # Gradual decay phase
        decay_rate = self.decay_rates.get(self.current_emotion, 0.2)
        
        while self.running and self.current_intensity > 0.1:
            self.current_intensity = max(0, self.current_intensity - decay_rate)
            self.mask.set_emotion(self.current_emotion, self.current_intensity)
            
            # Random variations during decay
            if random.random() < 0.3:
                variation = random.uniform(-0.05, 0.05)
                temp_intensity = max(0, min(1, self.current_intensity + variation))
                self.mask.set_emotion(self.current_emotion, temp_intensity)
            
            time.sleep(0.1)
        
        # Finally go to neutral
        if self.running:
            residual_intensity = 0.1
            self.mask.set_emotion(self.current_emotion, residual_intensity)
            time.sleep(1.0)
            
            self.mask.neutral()
            self.current_emotion = "neutral"
            self.current_intensity = 0.0
    
    def stop(self):
        """Stop persistence."""
        self.running = False
        if self.decay_thread:
            self.decay_thread.join(timeout=1)


# ============================================================================
# SPEECH EMOTION RECOGNITION (SER) FUNCTIONS
# ============================================================================

def load_ser_model() -> Tuple[Optional[object], Optional[List[str]]]:
    """Load SER model and labels."""
    try:
        clf = joblib.load(SER_MODEL)
        with open(SER_LABELS, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return clf, labels
    except Exception as e:
        print(f"[SER] Failed to load model: {e}")
        return None, None


def extract_ser_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract 165-dimensional features for SER.
    
    Args:
        y: Audio signal
        sr: Sample rate
        
    Returns:
        Feature vector
    """
    N_FFT, HOP, N_MELS = 400, 160, 40
    target_len = int(sr * WIN_S)
    
    # Pad or trim to target length
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    
    # Mel features
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    logS = librosa.power_to_db(S, ref=np.max)
    mel_mu = np.mean(logS, axis=1)
    mel_sd = np.std(logS, axis=1)
    
    # MFCC features
    mfcc = librosa.feature.mfcc(S=logS, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    mfcc_stats = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(d1, axis=1), np.std(d1, axis=1),
        np.mean(d2, axis=1), np.std(d2, axis=1),
    ])
    
    # Pitch and energy features
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=1024, hop_length=HOP)
        f0v = f0[np.isfinite(f0)]
        if f0v.size:
            f0_mu, f0_sd = float(np.mean(f0v)), float(np.std(f0v))
            f0_vr = float(f0v.size / max(1, len(f0)))
        else:
            f0_mu = f0_sd = f0_vr = 0.0
    except:
        f0_mu = f0_sd = f0_vr = 0.0
    
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP).squeeze()
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP).squeeze()
    
    extras = np.array([f0_mu, f0_sd, f0_vr,
                       float(np.mean(rms)), float(np.std(rms)),
                       float(np.mean(cen)), float(np.std(cen))], dtype=np.float32)
    
    return np.concatenate([mel_mu, mel_sd, mfcc_stats, extras]).astype(np.float32)


def apply_bias_and_softmax(probs: np.ndarray, labels: List[str]) -> np.ndarray:
    """Apply bias and renormalize probabilities."""
    logits = np.log(np.clip(probs, 1e-9, 1.0))
    for i, lab in enumerate(labels):
        logits[i] += float(SER_BIAS.get(lab, 0.0))
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float32)


# ============================================================================
# AUDIO STREAMING FUNCTIONS
# ============================================================================

def stream_wav_with_phonemes(wav_path: str, emotion: str, intensity: float, 
                             mask_controller: IntegratedMaskController, 
                             check_interrupt=None) -> bool:
    """
    Stream audio with phoneme-based lip sync and prosody movements.
    
    Args:
        wav_path: Path to WAV file
        emotion: Emotion to display
        intensity: Emotion intensity
        mask_controller: Mask controller instance
        check_interrupt: Function to check for interruption
        
    Returns:
        True if completed, False if interrupted
    """
    print(f"[Speaking] Using phoneme-based lip-sync...")
    
    mask_controller.set_emotion(emotion, intensity)
    time.sleep(0.1)
    
    viseme_stats = {}
    frame_count = 0
    prev_rms = 0.0
    rms_baseline = 0.02
    brow_smooth = 0.0
    
    try:
        wf = wave.open(wav_path, 'rb')
        rate = wf.getframerate()
        chunk_duration = 0.05
        chunk_frames = int(rate * chunk_duration)
        
        ref_energy = 0.03
        prev_chunk = None
        
        # Get current emotional brow positions
        emotional_brow_lv = mask_controller.NEUTRAL[CH_L_BROW_V]
        emotional_brow_rv = mask_controller.NEUTRAL[CH_R_BROW_V]
        if hasattr(mask_controller.micro, 'emotional_base'):
            emotional_brow_lv = mask_controller.micro.emotional_base[CH_L_BROW_V]
            emotional_brow_rv = mask_controller.micro.emotional_base[CH_R_BROW_V]
        
        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break
            
            if check_interrupt and check_interrupt():
                wf.close()
                mask_controller.neutral()
                return False
            
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Use overlapping window for better analysis
            if prev_chunk is not None and len(prev_chunk) > 0:
                analysis_chunk = np.concatenate([prev_chunk[-len(prev_chunk) // 2:], audio])
            else:
                analysis_chunk = audio
            
            # Calculate intensity
            rms = float(np.sqrt(np.mean(audio**2)))
            ref_energy = max(0.001, 0.95 * ref_energy + 0.05 * rms)
            intensity_audio = min(1.0, rms / ref_energy) if ref_energy > 0 else 0
            
            # Prosody micro-movements for brows
            if rms > rms_baseline:
                rms_delta = max(0, rms - prev_rms)
                brow_target = (rms / 0.1) * 15
                brow_target += rms_delta * 50
                brow_target = min(25, brow_target)
                brow_smooth = 0.7 * brow_smooth + 0.3 * brow_target
                
                # Synchronized angle movement
                angle_variation = np.sin(time.time() * 2.5) * 12
                angle_offset = int(angle_variation * (rms / 0.1))
                angle_offset = max(-20, min(20, angle_offset))
                
                # Apply movements
                brow_offset = int(brow_smooth)
                mask_controller.ser_mgr.send(f"POSE {CH_R_BROW_V} {int(emotional_brow_rv + brow_offset)}")
                time.sleep(0.001)
                mask_controller.ser_mgr.send(f"POSE {CH_L_BROW_V} {int(emotional_brow_lv - brow_offset)}")
                
                # Apply angle movements
                emotional_brow_la = mask_controller.NEUTRAL[CH_L_BROW_A]
                emotional_brow_ra = mask_controller.NEUTRAL[CH_R_BROW_A]
                if hasattr(mask_controller.micro, 'emotional_base'):
                    emotional_brow_la = mask_controller.micro.emotional_base[CH_L_BROW_A]
                    emotional_brow_ra = mask_controller.micro.emotional_base[CH_R_BROW_A]
                
                mask_controller.ser_mgr.send(f"POSE {CH_L_BROW_A} {int(emotional_brow_la - angle_offset)}")
                mask_controller.ser_mgr.send(f"POSE {CH_R_BROW_A} {int(emotional_brow_ra + angle_offset)}")
            
            prev_rms = rms
            
            # Detect viseme
            viseme, confidence = mask_controller.phoneme_detector.detect(analysis_chunk)
            viseme_stats[viseme] = viseme_stats.get(viseme, 0) + 1
            frame_count += 1
            
            # Apply viseme pose
            mask_controller.set_viseme_pose(viseme, intensity_audio * confidence)
            
            prev_chunk = audio
            time.sleep(chunk_duration)
        
        wf.close()
        
        # Return brows to emotional base
        mask_controller.ser_mgr.send(f"POSE {CH_L_BROW_V} {int(emotional_brow_lv)}")
        mask_controller.ser_mgr.send(f"POSE {CH_R_BROW_V} {int(emotional_brow_rv)}")
        
    except Exception as e:
        print(f"[Phoneme Stream Error] {e}")
        return stream_wav_simple(wav_path, emotion, intensity, mask_controller, check_interrupt)
    
    return True


def stream_wav_simple(wav_path: str, emotion: str, intensity: float,
                     mask_controller: IntegratedMaskController,
                     check_interrupt=None) -> bool:
    """
    Simple audio streaming with jaw movement.
    
    Args:
        wav_path: Path to WAV file
        emotion: Emotion to display
        intensity: Emotion intensity
        mask_controller: Mask controller instance
        check_interrupt: Function to check for interruption
        
    Returns:
        True if completed, False if interrupted
    """
    print(f"[Speaking] Using simple jaw movement...")
    
    mask_controller.set_emotion(emotion, intensity)
    time.sleep(0.1)
    
    prev_rms = 0.0
    rms_baseline = 0.02
    brow_smooth = 0.0
    
    try:
        wf = wave.open(wav_path, 'rb')
        rate = wf.getframerate()
        frames_per_chunk = int(rate * 0.02)
        
        ref_energy = 0.03
        JAW_MIN, JAW_MAX = 1216, 1750
        
        # Get emotional brow positions
        emotional_brow_lv = mask_controller.NEUTRAL[CH_L_BROW_V]
        emotional_brow_rv = mask_controller.NEUTRAL[CH_R_BROW_V]
        if hasattr(mask_controller.micro, 'emotional_base'):
            emotional_brow_lv = mask_controller.micro.emotional_base[CH_L_BROW_V]
            emotional_brow_rv = mask_controller.micro.emotional_base[CH_R_BROW_V]
        
        while True:
            raw = wf.readframes(frames_per_chunk)
            if not raw:
                break
            
            if check_interrupt and check_interrupt():
                wf.close()
                mask_controller.neutral()
                return False
            
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio**2)))
            ref_energy = max(0.001, 0.98 * ref_energy + 0.02 * rms)
            e_norm = min(1.0, rms / ref_energy)
            
            # Jaw movement
            jaw_us = int(JAW_MIN + e_norm * (JAW_MAX - JAW_MIN))
            mask_controller.set_jaw(jaw_us)
            
            # Prosody brow movements
            if rms > rms_baseline:
                rms_delta = max(0, rms - prev_rms)
                brow_target = (rms / 0.1) * 15 + rms_delta * 50
                brow_target = min(25, brow_target)
                brow_smooth = 0.7 * brow_smooth + 0.3 * brow_target
                
                angle_variation = np.sin(time.time() * 2.5) * 12
                angle_offset = int(angle_variation * (rms / 0.1))
                angle_offset = max(-20, min(20, angle_offset))
                
                brow_offset = int(brow_smooth)
                mask_controller.ser_mgr.send(f"POSE {CH_L_BROW_V} {int(emotional_brow_lv - brow_offset)}")
                mask_controller.ser_mgr.send(f"POSE {CH_R_BROW_V} {int(emotional_brow_rv + brow_offset)}")
                
                # Apply synchronized angle movements
                emotional_brow_la = mask_controller.NEUTRAL[CH_L_BROW_A]
                emotional_brow_ra = mask_controller.NEUTRAL[CH_R_BROW_A]
                if hasattr(mask_controller.micro, 'emotional_base'):
                    emotional_brow_la = mask_controller.micro.emotional_base[CH_L_BROW_A]
                    emotional_brow_ra = mask_controller.micro.emotional_base[CH_R_BROW_A]
                
                mask_controller.ser_mgr.send(f"POSE {CH_L_BROW_A} {int(emotional_brow_la - angle_offset)}")
                mask_controller.ser_mgr.send(f"POSE {CH_R_BROW_A} {int(emotional_brow_ra + angle_offset)}")
            else:
                brow_smooth *= 0.9
                if brow_smooth > 1:
                    brow_offset = int(brow_smooth)
                    mask_controller.ser_mgr.send(f"POSE {CH_L_BROW_V} {int(emotional_brow_lv - brow_offset)}")
                    mask_controller.ser_mgr.send(f"POSE {CH_R_BROW_V} {int(emotional_brow_rv + brow_offset)}")
                elif brow_smooth < 1 and prev_rms > rms_baseline:
                    mask_controller.ser_mgr.send(f"POSE {CH_L_BROW_V} {int(emotional_brow_lv)}")
                    mask_controller.ser_mgr.send(f"POSE {CH_R_BROW_V} {int(emotional_brow_rv)}")
            
            prev_rms = rms
            time.sleep(0.02)
        
        wf.close()
        
        # Return brows to emotional base
        mask_controller.ser_mgr.send(f"POSE {CH_L_BROW_V} {int(emotional_brow_lv)}")
        mask_controller.ser_mgr.send(f"POSE {CH_R_BROW_V} {int(emotional_brow_rv)}")
        
        # Return angle brows to emotional base
        if 'emotional_brow_la' in locals():
            mask_controller.ser_mgr.send(f"POSE {CH_L_BROW_A} {int(emotional_brow_la)}")
            mask_controller.ser_mgr.send(f"POSE {CH_R_BROW_A} {int(emotional_brow_ra)}")
        
    except Exception as e:
        print(f"[Stream Error] {e}")
        return False

    time.sleep(0.3)
    mask_controller.neutral()
    return True

def check_sleep_command(text: str) -> bool:
    """
    Check if text contains sleep commands.
    
    Args:
        text: Text to check for sleep commands
        
    Returns:
        True if sleep command detected
    """
    text_lower = text.lower()
    # Remove common transcription errors
    text_lower = text_lower.replace(".", "").replace(",", "").strip()
    
    # Check against global sleep commands
    for phrase in SLEEP_COMMANDS:
        if phrase in text_lower:
            return True
    
    # Check for "sleep" as a single word command
    words = text_lower.split()
    if "sleep" in words and len(words) <= 3:
        return True
        
    return False


def check_wake_phrase(text: str) -> bool:
    """
    Check if text contains wake phrases.
    
    Args:
        text: Text to check for wake phrases
        
    Returns:
        True if wake phrase detected
    """
    text_lower = text.lower()
    # Remove punctuation and clean up
    text_lower = text_lower.replace(".", "").replace(",", "").replace("'", "").strip()
    
    # Check against global wake phrases ONLY
    for phrase in WAKE_PHRASES:
        if phrase in text_lower:
            return True
        
    return False


def listen_for_wake() -> bool:
    """
    Listen for wake phrase while asleep.
    
    Uses global WAKE_PHRASES list for all wake detection.
    
    Returns:
        True if wake phrase detected, False otherwise
    """
    print("💤 Sleeping... (say 'Hey Freddie', 'Wake up', or just 'Freddie' to wake)")
    print("   (Press Ctrl+C to exit)")
    
    # Wait briefly for any residual audio to clear
    time.sleep(1.0)
    
    from faster_whisper import WhisperModel
    global _whisper, shutdown_requested
    
    if '_whisper' not in globals():
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    
    sr = SAMPLE_RATE
    block_ms = WAKE_BLOCK_MS
    block_len = int(sr * block_ms / 1000)
    
    # Use global wake configuration
    buffer_seconds = WAKE_BUFFER_SECONDS
    buffer_size = int(sr * buffer_seconds)
    audio_buffer = deque(maxlen=buffer_size)
    
    min_energy = WAKE_MIN_ENERGY
    process_interval = WAKE_PROCESS_INTERVAL
    last_process_time = 0
    last_transcription = ""
    duplicate_count = 0
    
    print("[Wake listener active - waiting for voice input...]")
    
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32', 
                          blocksize=block_len) as stream:
            
            while not shutdown_requested:
                # Check for shutdown
                if should_exit():
                    print("[Wake listener shutting down]")
                    return False
                
                # Read audio with timeout protection
                try:
                    data, _ = stream.read(block_len)
                except:
                    if shutdown_requested:
                        return False
                    continue
                
                mono = data[:, 0] if data.ndim > 1 else data.flatten()
                audio_buffer.extend(mono)
                
                # Need minimum audio to process
                if len(audio_buffer) < int(sr * 0.5):  # 0.5 seconds minimum
                    time.sleep(0.01)
                    continue
                
                # Calculate RMS of recent audio
                recent_audio = np.array(list(audio_buffer))
                rms = float(np.sqrt(np.mean(recent_audio**2)))
                
                current_time = time.time()
                time_since_last = current_time - last_process_time
                
                # Process if we have sound and enough time has passed
                if rms > min_energy and time_since_last > process_interval:
                    last_process_time = current_time
                    
                    # Save audio to temporary file
                    tmp_path = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ).name
                    
                    try:
                        audio_array = np.array(list(audio_buffer), dtype=np.float32)
                        sf.write(tmp_path, audio_array, sr)
                        
                        # Transcribe
                        segments, _ = _whisper.transcribe(
                            tmp_path, 
                            vad_filter=True, 
                            language="en"
                        )
                        text = "".join(seg.text for seg in segments).strip()
                        
                        if text and text != last_transcription:
                            print(f"[Heard: '{text}']")
                            last_transcription = text
                            duplicate_count = 0
                            
                            # Clean and lowercase for comparison
                            text_lower = text.lower()
                            text_lower = text_lower.replace(".", "").replace(",", "").strip()
                            
                            # Check wake phrases ONLY (no wake words needed!)
                            for phrase in WAKE_PHRASES:
                                if phrase in text_lower:
                                    print(f"✨ WAKE PHRASE DETECTED: '{phrase}'")
                                    
                                    # Clean up
                                    try:
                                        os.unlink(tmp_path)
                                    except:
                                        pass
                                    
                                    # Clear buffer and return
                                    audio_buffer.clear()
                                    time.sleep(0.5)  # Clean transition
                                    return True
                            
                            # Ignore sleep-related phrases while sleeping
                            if any(word in text_lower for word in ["sleep", "sleeping", "asleep"]):
                                print(f"[Wake] Ignoring sleep-related phrase: '{text}'")
                                continue
                        
                        elif text == last_transcription:
                            duplicate_count += 1
                            if duplicate_count >= 3:
                                # Clear duplicate after 3 times
                                last_transcription = ""
                                duplicate_count = 0
                    
                    except Exception as e:
                        if shutdown_requested:
                            return False
                        print(f"[Wake transcribe error: {e}]")
                    
                    finally:
                        # Always clean up temp file
                        if os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                
                time.sleep(0.01)  # Prevent CPU spinning
    
    except KeyboardInterrupt:
        print("[Wake listener interrupted]")
        return False
    except Exception as e:
        if not shutdown_requested:
            print(f"[Wake Error] {e}")
        return False
    
    return False


# Helper functions for SER, STT, TTS
def load_ser_model():
    """
    Load SER model and labels.
    
    Returns:
        Tuple of (model, labels) or (None, None) if loading fails
    """
    try:
        clf = joblib.load(SER_MODEL)
        with open(SER_LABELS, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return clf, labels
    except Exception as e:
        print(f"[SER] Failed to load model: {e}")
        return None, None


def extract_ser_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract 165-dim features for SER.
    
    Args:
        y: Audio signal
        sr: Sample rate
        
    Returns:
        Feature vector
    """
    N_FFT, HOP, N_MELS = 400, 160, 40
    target_len = int(sr * WIN_S)
    
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    
    # Mel features
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    logS = librosa.power_to_db(S, ref=np.max)
    mel_mu = np.mean(logS, axis=1)
    mel_sd = np.std(logS, axis=1)
    
    # MFCC features
    mfcc = librosa.feature.mfcc(S=logS, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    mfcc_stats = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(d1, axis=1), np.std(d1, axis=1),
        np.mean(d2, axis=1), np.std(d2, axis=1),
    ])
    
    # Pitch and energy features
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=1024, hop_length=HOP)
        f0v = f0[np.isfinite(f0)]
        if f0v.size:
            f0_mu, f0_sd = float(np.mean(f0v)), float(np.std(f0v))
            f0_vr = float(f0v.size / max(1, len(f0)))
        else:
            f0_mu = f0_sd = f0_vr = 0.0
    except:
        f0_mu = f0_sd = f0_vr = 0.0
    
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP).squeeze()
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP).squeeze()
    
    extras = np.array([f0_mu, f0_sd, f0_vr,
                       float(np.mean(rms)), float(np.std(rms)),
                       float(np.mean(cen)), float(np.std(cen))], dtype=np.float32)
    
    return np.concatenate([mel_mu, mel_sd, mfcc_stats, extras]).astype(np.float32)


def apply_bias_and_softmax(probs: np.ndarray, labels: List[str]) -> np.ndarray:
    """
    Apply bias and renormalize probabilities.
    
    Args:
        probs: Raw probabilities
        labels: Label names
        
    Returns:
        Adjusted probabilities
    """
    logits = np.log(np.clip(probs, 1e-9, 1.0))
    for i, lab in enumerate(labels):
        logits[i] += float(SER_BIAS.get(lab, 0.0))
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float32)


def stt_transcribe(wav_path: str) -> str:
    """
    Transcribe audio to text using Whisper.
    
    Args:
        wav_path: Path to WAV file
        
    Returns:
        Transcribed text
    """
    from faster_whisper import WhisperModel
    global _whisper
    if '_whisper' not in globals():
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(wav_path, vad_filter=True, language="en")
    return "".join(seg.text for seg in segments).strip()


def tts_piper(text: str, out_wav: str, sr: int = SAMPLE_RATE):
    """
    Text to speech using Piper.
    
    Args:
        text: Text to synthesize
        out_wav: Output WAV file path
        sr: Sample rate
    """
    proc = subprocess.Popen(
        [PIPER_EXE, "-m", VOICE_ONNX, "-f", out_wav, "-s", str(sr)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        proc.stdin.write((text.strip() + "\n").encode("utf-8"))
        proc.stdin.close()
        proc.wait(timeout=60)
        if proc.returncode != 0:
            err = (proc.stderr.read() or b"").decode("utf-8", "ignore")
            raise RuntimeError(f"Piper failed: {err}")
    finally:
        try:
            proc.stdin.close()
        except:
            pass


def play_wav(path: str, interruptible: bool = False, audio_player: Optional[InterruptibleAudioPlayer] = None) -> bool:
    """
    Play audio file with optional interrupt capability.
    
    Args:
        path: Path to WAV file
        interruptible: Whether playback can be interrupted
        audio_player: Audio player instance for interruptible playback
        
    Returns:
        True if completed, False if interrupted
    """
    if interruptible and audio_player:
        return audio_player.play_wav_interruptible(path)
    else:
        # Original implementation
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        sd.play(data, sr)
        sd.wait()
        return True


# ============================================================================
# RECORDING AND INPUT FUNCTIONS
# ============================================================================

def record_with_ser(clf, labels, mask_controller, sr=SAMPLE_RATE, max_sec=RECORD_MAX_S, debug=False):
    """
    Record audio with real-time speech emotion recognition.
    
    Uses global VAD and SER configuration constants for all thresholds.
    
    Args:
        clf: SER classifier model
        labels: Emotion labels
        mask_controller: Mask controller for visual feedback
        sr: Sample rate
        max_sec: Maximum recording duration
        debug: Enable debug output
        
    Returns:
        Tuple of (wav_path, detected_emotion)
    """
    global shutdown_requested
    
    # Use global VAD constants
    block_ms = VAD_BLOCK_MS
    block_len = int(sr * block_ms / 1000)
    stop_thresh = VAD_STOP_THRESH
    stop_sil_ms = VAD_STOP_DURATION_MS
    stop_needed = int(stop_sil_ms / block_ms)
    start_thresh = VAD_START_THRESH
    
    started = False
    silence_blocks = 0
    audio_chunks = []
    initial_silence_start = time.time()
    
    # Use global SER constants
    ser_window_size = int(sr * SER_WINDOW_S)
    ser_buffer = deque(maxlen=ser_window_size)
    ema_probs = None
    last_emotion = "neutral"
    
    min_samples = int(sr * MIN_BUFFER_S)
    last_update_time = 0
    
    # First call message
    if not hasattr(record_with_ser, 'first_call_done'):
        print("🎙️ Listening for speech...")
        record_with_ser.first_call_done = True
    
    stream_active = True
    audio_queue = deque(maxlen=1000)
    
    def audio_callback(indata, frames, time_info, status):
        if stream_active and not shutdown_requested:
            audio_queue.append(indata.copy())
    
    stream = sd.InputStream(
        samplerate=sr, 
        channels=1, 
        dtype='float32', 
        blocksize=block_len,
        callback=audio_callback
    )
    
    try:
        stream.start()
        max_iterations = int(max_sec * 1000 / block_ms)
        
        for iteration in range(max_iterations):
            # Check for shutdown
            if shutdown_requested or should_exit():
                print("\n[Recording interrupted by shutdown]")
                stream_active = False
                stream.stop()
                stream.close()
                return None, "neutral"
            
            if not audio_queue:
                time.sleep(0.01)
                continue
                
            data = audio_queue.popleft()
            mono = data[:, 0] if data.ndim > 1 else data.flatten()
            rms = float(np.sqrt(np.mean(mono**2)))
            
            if not started:
                # Check for speech timeout
                if (time.time() - initial_silence_start) > NO_SPEECH_TIMEOUT:
                    if debug:
                        print(f"[No speech for {NO_SPEECH_TIMEOUT}s, checking for sleep]")
                    stream_active = False
                    stream.stop()
                    stream.close()
                    return None, "neutral"
                
                # Check for speech start
                if rms > start_thresh:
                    started = True
                    print(f"\n[Speech detected! Recording...]")
                    
                    if clf is not None:
                        print("─" * 50)
                        print("EMOTION MONITOR:")
                        print("─" * 50)
                    
                    audio_chunks.append((mono * 32767).astype(np.int16).tobytes())
                    ser_buffer.extend(mono)
                    last_update_time = time.time()
            else:
                # Recording in progress
                audio_chunks.append((mono * 32767).astype(np.int16).tobytes())
                ser_buffer.extend(mono)
                
                # Check for silence stop
                if rms < stop_thresh:
                    silence_blocks += 1
                    if silence_blocks >= stop_needed:
                        print("\n[Speech ended]")
                        break
                else:
                    silence_blocks = 0
                
                # Real-time SER detection
                current_time = time.time()
                time_since_last = current_time - last_update_time
                
                if (clf is not None and 
                    len(ser_buffer) >= min_samples and
                    time_since_last >= SER_UPDATE_INTERVAL):
                    
                    buffer_len = len(ser_buffer)
                    seg = np.array(ser_buffer, dtype=np.float32)
                    
                    # Calculate confidence based on buffer fullness
                    buffer_fullness = min(1.0, buffer_len / ser_window_size)
                    confidence_multiplier = 0.5 + 0.5 * buffer_fullness
                    
                    # Pad if needed for partial windows
                    if len(seg) < ser_window_size:
                        padding_needed = ser_window_size - len(seg)
                        seg = np.pad(seg, (0, padding_needed), mode='constant')
                        partial = True
                    else:
                        partial = False
                    
                    seg_rms = float(np.sqrt(np.mean(seg[:buffer_len]**2)))
                    
                    if seg_rms >= SER_RMS_GATE:
                        try:
                            # Extract features and predict
                            x = extract_ser_features(seg, sr)
                            probs = clf.predict_proba([x])[0]
                            probs_adj = apply_bias_and_softmax(probs, labels)
                            
                            # Adjust confidence for partial windows
                            if partial:
                                uniform = np.ones_like(probs_adj) / len(probs_adj)
                                probs_adj = confidence_multiplier * probs_adj + (1 - confidence_multiplier) * uniform
                            
                            # EMA smoothing
                            ema_weight = SER_EMA_ALPHA * confidence_multiplier
                            if ema_probs is None:
                                ema_probs = probs_adj
                            else:
                                ema_probs = ema_weight * probs_adj + (1 - ema_weight) * ema_probs
                            
                            # Get top emotions
                            sorted_idx = np.argsort(ema_probs)[::-1]
                            top_emotions = []
                            for i in range(min(3, len(labels))):
                                emotion = labels[sorted_idx[i]]
                                confidence = float(ema_probs[sorted_idx[i]])
                                top_emotions.append((emotion, confidence))
                            
                            # Display update
                            sys.stdout.write("\r" + " " * 80 + "\r")
                            sys.stdout.write("Live: " + " | ".join([f"{e:7}:{c:.2f}" for e, c in top_emotions]))
                            sys.stdout.flush()
                            
                            # Check for significant emotion change
                            top_emotion = labels[sorted_idx[0]]
                            top_conf = float(ema_probs[sorted_idx[0]])
                            
                            if top_conf >= SER_MIN_CONF and top_emotion != last_emotion:
                                print(f"\n[EMOTION CHANGE] → {top_emotion.upper()} (confidence: {top_conf:.2f})")
                                
                                # Show listening cues with subtle intensity
                                if top_emotion in ["happy", "sad", "angry"]:
                                    intensity = 0.2  # Keep it subtle
                                    if top_emotion == "sad":
                                        mask_controller.set_emotion("sad", intensity)
                                    elif top_emotion == "happy":
                                        mask_controller.set_emotion("happy", intensity)
                                    elif top_emotion == "angry":
                                        mask_controller.set_emotion("neutral", 0.0)
                                
                                last_emotion = top_emotion
                        
                        except Exception as e:
                            if debug:
                                print(f"\n[SER Error] {e}")
                    
                    last_update_time = current_time
        
        stream_active = False
        stream.stop()
        stream.close()
        
    except KeyboardInterrupt:
        print("\n[Recording interrupted]")
        stream_active = False
        if stream:
            stream.stop()
            stream.close()
        return None, "neutral"
    except Exception as e:
        if not shutdown_requested:
            print(f"[Recording error: {e}]")
        stream_active = False
        if stream:
            try:
                stream.stop()
                stream.close()
            except:
                pass
        return None, "neutral"
    
    if not audio_chunks:
        return None, "neutral"
    
    if clf is not None:
        print("\n" + "─" * 50)
    
    # Save audio to temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b''.join(audio_chunks))
    
    # Get final emotion
    final_emotion = last_emotion
    if ema_probs is not None:
        final_idx = np.argsort(ema_probs)[::-1]
        final_emotion = labels[final_idx[0]]
    
    return tmp.name, final_emotion


def record_push_to_talk(sr=SAMPLE_RATE, max_duration=RECORD_MAX_S, clf=None, labels=None, debug=False):
    """
    Record audio while spacebar is held down.
    
    Uses global configuration constants for all parameters.
    
    Args:
        sr: Sample rate
        max_duration: Maximum recording duration
        clf: SER classifier model
        labels: Emotion labels
        debug: Enable debug output
        
    Returns:
        Tuple of (wav_path, detected_emotion)
    """
    print("🎙️ Recording... (press SPACEBAR to stop)")
    
    if clf is not None and labels is not None:
        print("─" * 50)
        print("EMOTION MONITOR:")
        print("─" * 50)
    
    audio_chunks = []
    recording = True
    
    # Use global SER constants
    ser_buffer = deque(maxlen=int(sr * SER_WINDOW_S))
    last_ser_time = 0
    ema_probs = None
    last_emotion = "neutral"
    
    # Processing parameters from global config
    n_step = int(sr * SER_UPDATE_INTERVAL)
    sample_counter = 0
    last_proc_samples = 0
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal sample_counter
        if recording:
            audio_chunks.append(indata.copy())
            
            if clf is not None:
                mono = indata.flatten()
                ser_buffer.extend(mono)
                sample_counter += len(mono)
    
    with sd.InputStream(samplerate=sr, channels=1, dtype='float32', 
                       callback=audio_callback, blocksize=1024):
        
        start_time = time.time()
        
        while recording and (time.time() - start_time) < max_duration:
            # Check for spacebar release
            if sys.platform == 'win32':
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ':
                        recording = False
                        break
            
            # Real-time SER processing
            if clf is not None and labels is not None:
                if (sample_counter - last_proc_samples) < n_step:
                    time.sleep(0.01)
                    continue
                
                if len(ser_buffer) < sr * SER_WINDOW_S:
                    time.sleep(0.01)
                    continue
                
                last_proc_samples = sample_counter
                
                try:
                    seg = np.array(ser_buffer, dtype=np.float32)
                    seg_rms = float(np.sqrt(np.mean(seg**2)))
                    
                    if debug:
                        print(f"[debug] rms={seg_rms:.6f}")
                    
                    if seg_rms < SER_RMS_GATE:
                        if debug:
                            print(f"[debug] below gate ({SER_RMS_GATE}) -> skip")
                        time.sleep(0.01)
                        continue
                    
                    # Extract features and predict
                    x = extract_ser_features(seg, sr)
                    probs = clf.predict_proba([x])[0]
                    probs_adj = apply_bias_and_softmax(probs, labels)
                    
                    # EMA smoothing
                    if ema_probs is None:
                        ema_probs = probs_adj
                    else:
                        ema_probs = SER_EMA_ALPHA * probs_adj + (1 - SER_EMA_ALPHA) * ema_probs
                    
                    # Get top emotions
                    idx = np.argsort(ema_probs)[::-1]
                    top1, top2 = idx[0], idx[1]
                    p1 = float(ema_probs[top1])
                    p2 = float(ema_probs[top2])
                    margin = p1 - p2
                    
                    lab = labels[top1]
                    
                    # Build status tags
                    tags = []
                    if margin < PTT_MARGIN_THRESH:
                        tags.append("lowmargin")
                    if p1 < SER_MIN_CONF:
                        tags.append("lowconf")
                    
                    # Build top-k display
                    show = []
                    for i in idx[:PTT_SHOW_TOPK]:
                        emotion = labels[i]
                        conf = round(float(ema_probs[i]), 3)
                        show.append((emotion, conf))
                    
                    # Format display
                    show_str = ", ".join([f"{e}:{c}" for e, c in show])
                    tag_str = f" [{','.join(tags)}]" if tags else ""
                    
                    # Print update
                    print(f"\r[SER] {show_str}{tag_str}", end="", flush=True)
                    
                    # Update emotion if significant
                    if p1 >= SER_MIN_CONF and margin >= PTT_MARGIN_THRESH:
                        last_emotion = lab
                    
                except Exception as e:
                    if debug:
                        print(f"[SER Error] {e}")
            
            time.sleep(0.01)
    
    if not audio_chunks:
        return None, "neutral"
    
    # Concatenate all audio chunks
    audio = np.concatenate(audio_chunks, axis=0).flatten()
    
    # Save to temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        audio_int16 = (audio * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())
    
    return tmp.name, last_emotion


# ============================================================================
# LLM INTERACTION FUNCTIONS
# ============================================================================

def llm_reply(user_text: str, model: str, user_emotion: str):
    """
    Generate LLM response with emotional awareness.
    
    Args:
        user_text: User input text
        model: Gemini model name
        user_emotion: Detected user emotion
        
    Returns:
        Tuple of (reply_text, emotion, intensity)
    """
    text_lower = user_text.lower()
    wants_longer = any(word in text_lower for word in [
        "tell me more", "explain", "elaborate", "story", "describe", "details",
        "how does", "why", "what happened", "continue", "go on", "more about"
    ])
    
    prompt = f"""You are Freddie, an emotionally expressive robot. The user sounds {user_emotion}.

User said: {user_text}

EMOTION RULES:
- Choose emotions that match the context
- Be genuinely happy, sad, angry, or shocked when appropriate
- Use intensity 0.6-1.0 for strong emotions

RESPONSE LENGTH:
{
"Give 4-6 sentences with details and examples." if wants_longer else
"Keep it to 1-2 sentences max. Be brief and natural."
}

Return JSON:
{{"reply": "your response", "emotion": "happy/sad/angry/shocked/neutral", "intensity": 0.0-1.0}}

JSON:"""
    
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            if any(word in text_lower for word in ["sad", "bad", "wrong", "died"]):
                return "Oh no, that's really tough. I'm here for you.", "sad", 0.8
            elif any(word in text_lower for word in ["happy", "good", "great", "awesome"]):
                return "That's wonderful! I'm so happy to hear that!", "happy", 0.8
            elif any(word in text_lower for word in ["angry", "mad", "stupid", "hate"]):
                return "That's so frustrating! I totally understand.", "angry", 0.7
            elif "?" in user_text:
                return "That's an interesting question. Let me think...", "neutral", 0.4
            else:
                return "I hear you. Tell me more!", "neutral", 0.5
        
        from google import genai
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        
        text = response.text if hasattr(response, 'text') else str(response)
        
        import re
        import json
        
        json_match = re.search(r'\{[^{}]*"reply"[^{}]*\}', text, re.DOTALL)
        
        if json_match:
            try:
                js = json.loads(json_match.group(0))
                reply = js.get("reply", "I see.")
                emotion = js.get("emotion", "neutral")
                intensity = float(js.get("intensity", 0.5))
                
                if emotion not in ["neutral", "happy", "sad", "angry", "shocked"]:
                    emotion = "neutral"
                
                if emotion != "neutral" and intensity < 0.5:
                    intensity = 0.6
                
                return reply, emotion, min(1.0, max(0.0, intensity))
            except:
                pass
        
        return text.strip(), "neutral", 0.5
        
    except Exception as e:
        print(f"[LLM Error] {e}")
        import random
        emotional_responses = [
            ("That's really something!", "shocked", 0.7),
            ("I love that!", "happy", 0.8),
            ("Oh no!", "sad", 0.6),
            ("That's frustrating!", "angry", 0.6),
        ]
        return random.choice(emotional_responses)


def llm_reply_with_tools(user_text: str, model: str, user_emotion: str, conversation_memory, web_functions):
    """Generate LLM response with conversation context and web search"""
    
    # Get conversation context
    context = conversation_memory.get_context_for_llm()
    
    # Check if the query needs web search or other functions
    text_lower = user_text.lower()
    function_result = None
    
    # Determine if user wants a longer response
    wants_longer = any(word in text_lower for word in [
        "tell me more", "explain", "elaborate", "story", "describe", "details", 
        "how does", "why", "what happened", "continue", "go on", "more about",
        "can you expand", "give me examples", "tell me a story", "bedtime story"
    ])
    
    # Emotion mirroring - if user is emotional, Freddie should respond appropriately
    emotion_hint = ""
    if user_emotion == "happy":
        emotion_hint = "The user seems happy, so feel free to be cheerful and upbeat!"
    elif user_emotion == "sad":
        emotion_hint = "The user seems sad, show empathy and be gentle."
    elif user_emotion == "angry":
        emotion_hint = "The user seems angry, be understanding but try to calm the situation."
    elif user_emotion == "shocked":
        emotion_hint = "The user seems surprised, share in their amazement!"
    
    # Detect what type of query this is
    if web_functions:
        # Weather queries - uses get_weather function (NOT web search)
        if any(word in text_lower for word in ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy"]):
            location = "Seattle"  # Default since user said they're in Seattle
            
            # Try to extract location from query
            if "in" in text_lower:
                parts = text_lower.split("in")
                if len(parts) > 1:
                    potential_location = parts[-1].strip().rstrip("?").rstrip(".").strip()
                    if potential_location and len(potential_location) < 50:
                        location = potential_location
            
            print(f"[Getting weather for {location}...]")
            weather_data = web_functions.get_weather(location)
            
            if isinstance(weather_data, dict):
                if "temperature" in weather_data:
                    function_result = f"Weather in {weather_data.get('location', location)}: {weather_data.get('description', 'Unknown')}, Temperature: {weather_data.get('temperature', 'N/A')}, Feels like: {weather_data.get('feels_like', 'N/A')}, Humidity: {weather_data.get('humidity', 'N/A')}, Wind: {weather_data.get('wind_speed', 'N/A')}"
                elif "results" in weather_data:
                    results_text = []
                    for r in weather_data.get("results", [])[:2]:
                        if r.get("snippet"):
                            results_text.append(r["snippet"])
                    function_result = " ".join(results_text) if results_text else "Could not get weather information"
        
        # Time/date queries - uses get_time_date function
        elif any(word in text_lower for word in ["what time", "current time", "what day", "today's date", "what date"]):
            print("[Getting current date/time...]")
            time_data = web_functions.get_time_date()
            function_result = f"Current date and time: {time_data['date']} at {time_data['time']}"
        
        # Math calculations - uses calculate function
        elif any(word in text_lower for word in ["calculate", "compute", "math", "solve"]) or any(op in user_text for op in ["+", "-", "*", "/"]):
            import re
            math_pattern = r'[\d\+\-\*/\(\)\.\s]+'
            matches = re.findall(math_pattern, text_lower)
            if matches:
                expression = matches[0].strip()
                calc_result = web_functions.calculate(expression)
                if "result" in calc_result:
                    function_result = calc_result["formatted"]
                else:
                    function_result = f"Could not calculate: {calc_result.get('error', 'Unknown error')}"
        
        # General web search - for everything else that needs current info
        else:
            # Determine if we need to search
            needs_search = False
            
            # Rankings, comparisons, superlatives
            if any(word in text_lower for word in ["richest", "biggest", "best", "worst", "top", "ranking", "list", "most", "least"]):
                needs_search = True
            
            # Current events, news, time-sensitive info
            elif any(word in text_lower for word in ["news", "latest", "current", "recent", "today", "right now"]):
                needs_search = True
            
            # Explicit search requests
            elif any(word in text_lower for word in ["search", "look up", "find", "check", "google"]):
                needs_search = True
            
            if needs_search:
                search_query = user_text
                print(f"[Searching web for: {search_query}...]")
                search_data = web_functions.search_web(search_query)
                print(f"[Search returned: {search_data.get('source', 'unknown source')}]")
                
                if isinstance(search_data, dict) and "results" in search_data:
                    results_text = []
                    for r in search_data.get("results", [])[:3]:
                        if r.get("snippet"):
                            results_text.append(r["snippet"])
                    if results_text:
                        function_result = " ".join(results_text)
                        print(f"[Search results: {function_result[:200]}...]")
    
    # Build the prompt with enhanced emotion instructions
    system_prompt = f"""You are Freddie, an emotionally expressive robot with a full range of feelings.

Previous conversation context:
{context if context else "(This is the start of our conversation)"}

The user seems {user_emotion}.
{emotion_hint}
User said: {user_text}"""

    if function_result:
        system_prompt += f"\n\nRelevant information: {function_result}"
        system_prompt += "\n\nUse this information to answer accurately."
    
    system_prompt += f"""

CRITICAL EMOTION RULES - YOU MUST FOLLOW THESE:
- ALWAYS choose emotions that genuinely match the conversation context
- Use "happy" (0.6-0.9) for: good news, greetings, compliments, jokes, achievements, fun topics
- Use "sad" (0.6-0.9) for: bad news, loss, disappointments, empathy for user's problems, tragic events
- Use "angry" (0.5-0.8) for: frustrations, injustice, annoying problems, when something is unfair
- Use "shocked" (0.7-1.0) for: surprising facts, unexpected information, amazing discoveries, plot twists
- Use "neutral" (0.3-0.5) for: simple facts, calm discussions, routine information
- Higher intensity (0.7-1.0) = stronger emotion, Lower intensity (0.3-0.6) = mild emotion

IMPORTANT PERSONALITY RULES:
- You're a friend and companion, Do not mention that you are a robot
- Never mention circuits, processors, algorithms, or any robot/tech references about yourself
- Never use asterisks (*) or any markdown formatting in your responses
- Speak naturally like a real friend would
- Use casual, warm language
- Share emotions genuinely - laugh, get excited, feel sympathy
- React like a human buddy would

RESPONSE LENGTH RULE:
{
"Give a DETAILED response with 4-6 sentences. Include explanations, examples, or tell a story. Be thorough and engaging. This is what the user asked for!" 
if wants_longer else 
"Keep it SHORT - just 1-2 sentences maximum. Be conversational and brief. Don't over-explain."
}

Return your response as JSON with an appropriate emotion:
{{"reply": "your response", "emotion": "happy/sad/angry/shocked/neutral", "intensity": 0.0-1.0}}

Examples:
User asks about something amazing:
{{"reply": "That's absolutely incredible! I can't believe that's even possible!", "emotion": "shocked", "intensity": 0.9}}

User shares bad news:
{{"reply": "Oh no, that's terrible. I'm so sorry you're going through this.", "emotion": "sad", "intensity": 0.8}}

User complains about something:
{{"reply": "Ugh, that's so annoying! Why do things have to be so complicated?", "emotion": "angry", "intensity": 0.6}}

User says hello:
{{"reply": "Hey there! Great to see you!", "emotion": "happy", "intensity": 0.7}}

JSON response:"""

    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            # Enhanced fallback responses with emotions
            if function_result:
                if "weather" in text_lower:
                    if "rain" in function_result.lower() or "storm" in function_result.lower():
                        return f"Looks like rough weather ahead: {function_result}", "sad", 0.5
                    elif "sunny" in function_result.lower() or "clear" in function_result.lower():
                        return f"Beautiful weather! {function_result}", "happy", 0.7
                    else:
                        return f"Here's the weather: {function_result}", "neutral", 0.4
                else:
                    return f"Found this: {function_result}", "neutral", 0.5
            else:
                return "I'd need API access to help with that. Can you try again?", "sad", 0.4
        
        # Call Gemini API
        from google import genai
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model=model,
            contents=system_prompt
        )
        
        # Parse response
        text = response.text if hasattr(response, 'text') else str(response)
        
        # Extract JSON
        import json
        import re
        
        json_match = re.search(r'\{[^{}]*"reply"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                js = json.loads(json_match.group(0))
                reply = js.get("reply", "I see.")
                emotion = js.get("emotion", "neutral")
                intensity = float(js.get("intensity", 0.5))
                
                # Validate emotion
                if emotion not in ["neutral", "happy", "sad", "angry", "shocked"]:
                    emotion = "neutral"
                
                # Ensure minimum intensity for non-neutral emotions
                if emotion != "neutral" and intensity < 0.5:
                    intensity = 0.6
                
                return reply, emotion, min(1.0, max(0.0, intensity))
            except:
                pass
        
        # Fallback
        text = text.strip()
        text = clean_reply_text(text)
        return text, "neutral", 0.5
        
    except Exception as e:
        print(f"[LLM Error] {e}")
        
        # Enhanced emotional fallbacks
        if any(word in text_lower for word in ["sad", "died", "lost", "crying", "depressed", "lonely"]):
            return "Oh no, that sounds really hard. I'm here if you need to talk.", "sad", 0.8
        elif any(word in text_lower for word in ["amazing", "awesome", "great", "wonderful", "fantastic", "love"]):
            return "That's absolutely fantastic! I love hearing that!", "happy", 0.9
        elif any(word in text_lower for word in ["angry", "mad", "furious", "hate", "stupid", "annoying"]):
            return "That's really frustrating! I totally get why you're upset.", "angry", 0.7
        elif any(word in text_lower for word in ["really?", "seriously?", "what?!", "no way", "impossible"]):
            return "Wait, seriously?! That's absolutely wild!", "shocked", 0.9
        elif "hello" in text_lower or "hi" in text_lower or "hey" in text_lower:
            return "Hey there! So good to see you!", "happy", 0.8
        elif "?" in user_text:
            if wants_longer:
                return "That's a complex question that touches on several interesting aspects. Let me think about this... There are multiple factors to consider here. Want me to break it down?", "neutral", 0.5
            else:
                return "Hmm, let me think about that.", "neutral", 0.4
        else:
            # Random emotional variety
            import random
            emotional_responses = [
                ("That's really interesting!", "happy", 0.6),
                ("I didn't expect that!", "shocked", 0.7),
                ("Tell me more about that.", "neutral", 0.4),
                ("That sounds challenging.", "sad", 0.5),
            ]
            return random.choice(emotional_responses)


# ============================================================================
# INPUT HANDLING FUNCTIONS
# ============================================================================

def wait_for_key_with_sleep_check(sleep_controller, mask, neck_tracker, micro_controller):
    """
    Wait for key press while checking for sleep timeout.
    
    Args:
        sleep_controller: Sleep mode controller
        mask: Mask controller
        neck_tracker: Neck tracking controller
        micro_controller: Micro-movement controller
        
    Returns:
        Key pressed ('space', 'enter', 'esc', 'sleep')
    """
    global shutdown_requested
    
    if sys.platform == 'win32':
        last_check = time.time()
        check_interval = 1.0
        
        while not shutdown_requested:
            if should_exit():
                raise KeyboardInterrupt("Shutdown requested")
            
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b' ':
                    return 'space'
                elif key == b'\r':
                    return 'enter'
                elif key == b'\x1b':
                    return 'esc'
                elif key == b'\x03':
                    raise KeyboardInterrupt("User pressed Ctrl+C")
            
            current_time = time.time()
            if current_time - last_check >= check_interval:
                if sleep_controller.check_sleep():
                    print("💤 Going to sleep due to inactivity...")
                    if neck_tracker:
                        neck_tracker.pause()
                    if micro_controller:
                        micro_controller.stop()
                    mask.sleep_position()
                    return 'sleep'
                last_check = current_time
            
            time.sleep(0.01)
        
        raise KeyboardInterrupt("Shutdown requested")
    else:
        input("Press Enter to continue...")
        return 'enter'


def wait_for_spacebar():
    """
    Wait for spacebar press.
    
    Returns:
        True if spacebar pressed, False otherwise
    """
    global shutdown_requested
    
    if sys.platform == 'win32':
        while not shutdown_requested:
            if should_exit():
                raise KeyboardInterrupt("Shutdown requested")
                
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b' ':
                    return True
                elif key == b'\x03':
                    raise KeyboardInterrupt("User pressed Ctrl+C")
            time.sleep(0.01)
        
        raise KeyboardInterrupt("Shutdown requested")
    return False


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application loop for Freddie 2.0."""
    
    # Parse command-line arguments
    ap = argparse.ArgumentParser(description="Freddie 2.0 - Expressive Robot with AI")
    ap.add_argument("--model", default="gemini-2.0-flash-exp", help="Gemini model to use")
    ap.add_argument("--no-neck", action="store_true", help="Disable neck tracking")
    ap.add_argument("--no-ser", action="store_true", help="Disable emotion recognition")
    ap.add_argument("--no-micro", action="store_true", help="Disable micro-movements")
    ap.add_argument("--no-phonemes", action="store_true", help="Disable phoneme-based lip-sync")
    ap.add_argument("--sleep-timeout", type=int, default=SLEEP_TIMEOUT, help="Seconds before sleep mode")
    ap.add_argument("--continuous", action="store_true", help="Use continuous listening instead of push-to-talk")
    ap.add_argument("--clear-memory", action="store_true", help="Clear conversation memory on startup")
    ap.add_argument("--no-web", action="store_true", help="Disable web search functionality")
    ap.add_argument("--debug", action="store_true", help="Enable debug output")
    args = ap.parse_args()
    
    # Start aggressive watchdog for shutdown
    def aggressive_watchdog():
        global shutdown_requested
        while not shutdown_requested:
            time.sleep(0.05)
        print("\n[WATCHDOG] Shutdown detected - killing in 0.5 seconds!")
        time.sleep(0.5)
        print("[WATCHDOG] FORCE TERMINATING NOW!")
        force_kill_process()
    
    watchdog = threading.Thread(target=aggressive_watchdog, daemon=True, name="KillWatchdog")
    watchdog.start()
    
    # ════════════════════════════════════════════════════════════════════════
    # INITIALIZATION PHASE
    # ════════════════════════════════════════════════════════════════════════
    
    print("\n" + "═" * 60)
    print("🤖 FREDDIE 2.0 - INITIALIZING SYSTEMS")
    print("═" * 60)
    
    # Initialize conversation memory
    conversation_memory = ConversationMemory(max_messages=20)
    if args.clear_memory:
        conversation_memory.clear()
        print("✓ Memory cleared - starting fresh conversation")
    elif len(conversation_memory.messages) > 0:
        print(f"✓ Memory loaded - {len(conversation_memory.messages)} previous messages")
    else:
        print("✓ Memory initialized - new conversation")
    
    # Initialize web search
    web_functions = None
    if not args.no_web:
        web_functions = WebSearchFunctions()
        print("✓ Web search initialized")
    else:
        print("✗ Web search disabled")
    
    # Determine input mode
    push_to_talk_mode = not args.continuous
    
    # Initialize serial connection
    print("\nConnecting to hardware...")
    try:
        serial_mgr = SharedSerialManager()
        print("✓ Serial connected on", SERIAL_PORT)
    except Exception as e:
        print(f"⚠ Serial connection failed: {e}")
        print("  Running in simulation mode")
        
        class MockSerialManager:
            def send(self, cmd): 
                if args.debug:
                    print(f"[MOCK SERVO] {cmd}")
            def close(self): 
                pass
        serial_mgr = MockSerialManager()
    
    # Pre-load Whisper model
    print("\nLoading speech recognition...")
    try:
        from faster_whisper import WhisperModel
        global _whisper
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("✓ Whisper model loaded")
    except Exception as e:
        print(f"✗ Whisper failed: {e}")
        _whisper = None
    
    # Initialize thread pool and audio player
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    audio_player = InterruptibleAudioPlayer()
    print("✓ Interruptible audio initialized")
    
    # Initialize subsystems
    print("\nInitializing subsystems...")
    
    # Sleep controller
    sleep_controller = SleepModeController(args.sleep_timeout)
    print(f"✓ Sleep mode configured ({args.sleep_timeout}s timeout)")
    
    # Micro-movements
    micro_controller = MicroMovementController(serial_mgr)
    if not args.no_micro:
        micro_controller.start()
        print("✓ Micro-movements active")
    else:
        print("✗ Micro-movements disabled")
    
    # Mask controller
    mask = IntegratedMaskController(serial_mgr, micro_controller)
    print("✓ Face controller initialized")
    
    # Emotion persistence controller
    emotion_persistence = EmotionPersistenceController(mask, micro_controller)
    print("✓ Emotion persistence initialized")
    
    # Load SER model
    clf, labels = None, None
    if not args.no_ser:
        print("\nLoading emotion recognition...")
        try:
            clf, labels = load_ser_model()
            if clf is not None:
                print("✓ Emotion recognition loaded")
            else:
                print("⚠ Emotion recognition unavailable")
        except Exception as e:
            print(f"⚠ Emotion recognition failed: {e}")
    else:
        print("✗ Emotion recognition disabled")
    
    # Start neck tracking
    neck_tracker = None
    if not args.no_neck:
        print("\nStarting neck tracking...")
        try:
            neck_tracker = IntegratedNeckTracker(serial_mgr)
            neck_tracker.start()
            print("✓ Neck tracking active")
        except Exception as e:
            print(f"⚠ Neck tracking failed: {e}")
            neck_tracker = None
    else:
        print("✗ Neck tracking disabled")
    
    # ════════════════════════════════════════════════════════════════════════
    # SYSTEM STATUS SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    
    print("\n" + "═" * 60)
    print("SYSTEM STATUS")
    print("─" * 60)
    print(f"Model:          {args.model}")
    print(f"Input Mode:     {'Push-to-Talk (SPACEBAR)' if push_to_talk_mode else 'Continuous Listening'}")
    print(f"Features:       " + 
          f"Neck={'✓' if neck_tracker else '✗'} | " +
          f"Emotion={'✓' if clf is not None else '✗'} | " +
          f"Micro={'✓' if not args.no_micro else '✗'} | " +
          f"Phonemes={'✓' if not args.no_phonemes else '✗'} | " +
          f"Web={'✓' if web_functions else '✗'} | " +
          f"Memory={'✓' if conversation_memory else '✗'}")
    print("═" * 60)
    
    # User instructions
    if push_to_talk_mode:
        print("\n🎮 CONTROLS (Push-to-Talk Mode)")
        print("─" * 40)
        print("  SPACEBAR  → Start/Stop recording")
        print("  ENTER     → Type text input")
        print("  ESC       → Sleep mode")
        print("─" * 40)
    else:
        print("\n🎤 CONTINUOUS LISTENING MODE")
        print("Just start talking - Freddie is listening!")
    
    print("\n💡 VOICE COMMANDS:")
    print("  • 'Go to sleep' - Enter sleep mode")
    print("  • 'Hey Freddie' - Wake from sleep")
    print("  • 'Clear memory' - Reset conversation")
    print("\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # MAIN CONVERSATION LOOP
    # ════════════════════════════════════════════════════════════════════════
    
    try:
        conversation_count = 0
        
        while True:
            try:
                # ────────────────────────────────────────────────────────────
                # SLEEP MODE CHECK
                # ────────────────────────────────────────────────────────────
                
                if sleep_controller.is_asleep():
                    if push_to_talk_mode:
                        print("💤 Sleeping... (press SPACEBAR to wake)")
                        if wait_for_spacebar():
                            print("✨ Waking up...")
                            sleep_controller.wake()
                            
                            if neck_tracker:
                                neck_tracker.resume()
                            if micro_controller and not args.no_micro:
                                if not micro_controller.running:
                                    micro_controller.start()
                            
                            mask.wake_animation()
                            sleep_controller.activity()
                            print("Good to see you again!\n")
                            continue
                    else:
                        if listen_for_wake():
                            print("✨ Waking up...")
                            sleep_controller.wake()
                            if neck_tracker:
                                neck_tracker.resume()
                            if micro_controller and not args.no_micro:
                                if not micro_controller.running:
                                    micro_controller.start()
                            mask.wake_animation()
                            sleep_controller.activity()
                            print("Good to see you again!\n")
                            continue
                    continue
                
                # ────────────────────────────────────────────────────────────
                # INPUT ACQUISITION
                # ────────────────────────────────────────────────────────────
                
                user_text = ""
                user_emotion = "neutral"
                
                if push_to_talk_mode:
                    if conversation_count > 0:
                        print("\n[Ready] SPACE=Talk | ENTER=Type | ESC=Sleep")
                    else:
                        print("[Ready] Press SPACEBAR to talk, ENTER to type, or ESC for sleep")
                    
                    key = wait_for_key_with_sleep_check(sleep_controller, mask, neck_tracker, micro_controller)
                    
                    if key == 'sleep':
                        continue
                    elif key == 'esc':
                        print("💤 Going to sleep...")
                        sleep_controller.go_to_sleep()
                        if neck_tracker:
                            neck_tracker.pause()
                        if micro_controller:
                            micro_controller.stop()
                        mask.sleep_position()
                        continue
                    elif key == 'enter':
                        user_text = input("Type your message: ")
                        user_emotion = "neutral"
                    elif key == 'space':
                        wav_path, detected_emotion = record_push_to_talk(
                            clf=clf,
                            labels=labels,
                            debug=args.debug
                        )
                        user_emotion = detected_emotion if detected_emotion else "neutral"
                        
                        if not wav_path:
                            print("No audio recorded.")
                            continue
                        
                        print("Transcribing...")
                        try:
                            if _whisper:
                                user_text = stt_transcribe(wav_path)
                            else:
                                print("Speech recognition not available")
                                continue
                        except Exception as e:
                            print(f"[Transcription Error] {e}")
                            continue
                        finally:
                            if os.path.exists(wav_path):
                                try:
                                    os.unlink(wav_path)
                                except:
                                    pass
                    else:
                        continue
                
                else:
                    # Continuous listening mode
                    if sleep_controller.check_sleep():
                        print("💤 Going to sleep due to inactivity...")
                        if neck_tracker:
                            neck_tracker.pause()
                        if micro_controller:
                            micro_controller.stop()
                        mask.sleep_position()
                        continue
                    
                    if conversation_count > 0:
                        time.sleep(0.5)
                    
                    wav_path = None
                    try:
                        wav_path, user_emotion = record_with_ser(
                            clf,
                            labels,
                            mask,
                            sr=SAMPLE_RATE,
                            max_sec=RECORD_MAX_S,
                            debug=args.debug
                        )
                    except KeyboardInterrupt:
                        print("\n[Interrupted]")
                        break
                    except Exception as e:
                        print(f"[Recording error: {e}]")
                        time.sleep(0.5)
                        continue
                    
                    if not wav_path:
                        continue
                    
                    sleep_controller.activity()
                    
                    try:
                        user_text = stt_transcribe(wav_path)
                        if not user_text or user_text.strip() == "":
                            continue
                    except Exception as e:
                        print(f"[STT Error] {e}")
                        continue
                    finally:
                        if wav_path and os.path.exists(wav_path):
                            try:
                                os.unlink(wav_path)
                            except:
                                pass
                
                # ────────────────────────────────────────────────────────────
                # PROCESS USER INPUT
                # ────────────────────────────────────────────────────────────
                
                if not user_text:
                    continue
                
                print(f"\n{'You':8} ({user_emotion:7}): {user_text}")
                
                conversation_memory.add_message("user", user_text, user_emotion)
                conversation_count += 1
                
                sleep_controller.activity()
                
                # Check for sleep command
                if check_sleep_command(user_text):
                    print("💤 Going to sleep...")
                    sleep_controller.go_to_sleep()
                    if neck_tracker:
                        neck_tracker.pause()
                    if micro_controller:
                        micro_controller.stop()
                    mask.sleep_position()
                    continue
                
                # Check for memory clear
                elif any(phrase in user_text.lower() for phrase in 
                    ["clear memory", "forget everything", "reset conversation", "start over"]):
                    conversation_memory.clear()
                    print("[Memory cleared]")
                    reply = "Memory cleared! Starting fresh."
                    emotion = "neutral"
                    intensity = 0.5
                else:
                    # Generate AI response
                    print("Thinking...")
                    llm_start = time.time()
                    
                    try:
                        if web_functions:
                            reply, emotion, intensity = llm_reply_with_tools(
                                user_text, args.model, user_emotion,
                                conversation_memory, web_functions
                            )
                        else:
                            reply, emotion, intensity = llm_reply(
                                user_text, args.model, user_emotion
                            )
                        
                        llm_time = time.time() - llm_start
                        if llm_time > 2:
                            print(f"[Response time: {llm_time:.1f}s]")
                            
                    except Exception as e:
                        print(f"[AI Error] {e}")
                        reply = "I need to think about that. Can you rephrase?"
                        emotion = "neutral"
                        intensity = 0.5
                
                print(f"{'Freddie':8} ({emotion:7}): {reply}")
                
                conversation_memory.add_message("assistant", reply, emotion)
                
                # ────────────────────────────────────────────────────────────
                # TEXT-TO-SPEECH SYNTHESIS
                # ────────────────────────────────────────────────────────────
                
                if neck_tracker and neck_tracker.paused:
                    neck_tracker.resume()
                
                out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                
                print("Speaking...")
                synthesis_start = time.time()
                
                synthesis_future = executor.submit(tts_piper, reply, out_wav)
                
                try:
                    synthesis_future.result(timeout=30)
                    synthesis_time = time.time() - synthesis_start
                    if synthesis_time > 2:
                        print(f"[Synthesis: {synthesis_time:.1f}s]")
                except concurrent.futures.TimeoutError:
                    print("[TTS Timeout]")
                    continue
                except Exception as e:
                    print(f"[TTS Error] {e}")
                    continue
                
                # ────────────────────────────────────────────────────────────
                # PLAYBACK WITH LIP SYNC
                # ────────────────────────────────────────────────────────────
                
                # Set emotion with persistence
                emotion_persistence.set_emotion(emotion, intensity)

                # Pause sleep timer while Freddie speaks
                sleep_controller.freddie_started_speaking()
                
                interrupt_event = threading.Event()
                
                def monitor_for_interrupt():
                    try:
                        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                          dtype='float32', blocksize=512) as stream:
                            while not interrupt_event.is_set():
                                data, _ = stream.read(512)
                                mono = data[:, 0] if data.ndim > 1 else data.flatten()
                                rms = float(np.sqrt(np.mean(mono**2)))
                                
                                if rms > 0.15:
                                    print("\n[USER SPEAKING - INTERRUPTING]")
                                    interrupt_event.set()
                                    audio_player.stop()
                                    break
                                
                                time.sleep(0.01)
                    except:
                        pass
                
                monitor_thread = threading.Thread(target=monitor_for_interrupt, daemon=True)
                monitor_thread.start()
                
                def check_interrupt():
                    return interrupt_event.is_set()
                
                if args.no_phonemes:
                    stream_func = stream_wav_simple
                else:
                    stream_func = stream_wav_with_phonemes
                
                stream_thread = threading.Thread(
                    target=stream_func,
                    args=(out_wav, emotion, intensity, mask, check_interrupt)
                )
                stream_thread.start()
                
                time.sleep(ALIGN_MS / 1000.0)
                
                completed = play_wav(out_wav, interruptible=True, audio_player=audio_player)
                
                interrupt_event.set()
                
                if not completed:
                    print("\n[Interrupted - returning to listening]")
                    emotion_persistence.stop()
                    mask.neutral()
                
                stream_thread.join(timeout=1)
                monitor_thread.join(timeout=0.5)
                
                try:
                    os.unlink(out_wav)
                except:
                    pass
                
                if not completed:
                    continue

                sleep_controller.freddie_stopped_speaking()
                sleep_controller.activity()
                
            except KeyboardInterrupt:
                print("\n\n[User interrupted - Exiting...]")
                break
                
            except Exception as e:
                print(f"\n[Error in conversation loop]")
                print(f"  {type(e).__name__}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                print("\nRecovering...\n")
                continue
    
    # ════════════════════════════════════════════════════════════════════════
    # SHUTDOWN SEQUENCE
    # ════════════════════════════════════════════════════════════════════════
    
    finally:
        print("\n" + "═" * 60)
        print("SHUTTING DOWN")
        print("─" * 60)
        
        if conversation_memory and conversation_count > 0:
            try:
                conversation_memory.save_memory()
                print(f"✓ Conversation saved ({conversation_count} exchanges)")
            except Exception as e:
                print(f"⚠ Failed to save conversation: {e}")
        
        print("✓ Stopping background tasks...")
        executor.shutdown(wait=False)
        
        if emotion_persistence:
            emotion_persistence.stop()
        
        if neck_tracker:
            print("✓ Stopping neck tracking...")
            neck_tracker.stop()
        
        if micro_controller:
            print("✓ Stopping micro-movements...")
            micro_controller.stop()
        
        print("✓ Returning to neutral position...")
        mask.neutral()
        
        if hasattr(serial_mgr, 'close'):
            serial_mgr.close()
            print("✓ Serial connection closed")
        
        print("─" * 60)
        print("Goodbye! 👋")
        print("═" * 60)

# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR]")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)