#!/usr/bin/env node
/**
 * Test script for the lip-sync API.
 *
 * Usage:
 *   node test-api.mjs --video https://example.com/video.mp4 --audio https://example.com/audio.mp3 --auto
 *   node test-api.mjs --video https://example.com/video.mp4 --audio https://example.com/audio.mp3 --refs refs.json --auto
 *   node test-api.mjs --video https://example.com/video.mp4 --audio https://example.com/audio.mp3 --bbox 100,50,200,250 --end 5000
 *
 * Options:
 *   --video      Input video URL (required)
 *   --audio      Audio URL for lip-sync (required)
 *   --output     Output video file (default: output.mp4)
 *   --auto       Auto-detect faces using InsightFace
 *   --refs       JSON file with character references for --auto mode
 *   --bbox       Face bounding box as x,y,w,h (for single face, alternative to --auto)
 *   --config     JSON file with face configuration (for multi-face)
 *   --start      Start time in ms (default: 0)
 *   --end        End time in ms (default: video duration for --auto, required for --bbox)
 *   --fps        Sample FPS for face detection (default: 3)
 *   --api        API base URL (default: http://localhost:8000)
 *   --no-enhance Disable CodeFormer enhancement
 *
 * Example refs.json (for character matching):
 * [
 *   {"id": "alice", "name": "Alice", "reference_image_url": "https://example.com/alice.jpg"},
 *   {"id": "bob", "name": "Bob", "reference_image_url": "https://example.com/bob.jpg"}
 * ]
 *
 * Example config.json (for manual face specification):
 * {
 *   "faces": [
 *     {"character_id": "alice", "bbox": [100, 50, 200, 250], "start_time_ms": 0, "end_time_ms": 5000},
 *     {"character_id": "bob", "bbox": [400, 60, 180, 220], "start_time_ms": 1000, "end_time_ms": 5000}
 *   ]
 * }
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';

// Check if a string is a URL
function isUrl(str) {
  return str.startsWith('http://') || str.startsWith('https://');
}

// Detect faces using InsightFace
async function detectFaces(apiBase, videoUrl, characters, sampleFps = 3, startTimeMs = null, endTimeMs = null) {
  console.log('Detecting faces with InsightFace...');
  console.log(`  Video URL: ${videoUrl}`);
  console.log(`  Sample FPS: ${sampleFps}`);
  console.log(`  Characters: ${characters.length}`);

  const requestBody = {
    video_url: videoUrl,
    sample_fps: sampleFps,
    characters: characters,
    similarity_threshold: 0.5,
  };

  if (startTimeMs !== null) {
    requestBody.start_time_ms = startTimeMs;
  }
  if (endTimeMs !== null) {
    requestBody.end_time_ms = endTimeMs;
  }

  const response = await fetch(`${apiBase}/detect-faces`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Face detection failed: ${response.status} - ${errorText}`);
  }

  const result = await response.json();
  console.log(`  Frame size: ${result.frame_width}x${result.frame_height}`);
  console.log(`  Video duration: ${result.video_duration_ms}ms`);
  console.log(`  Sampled ${result.frames.length} frame(s)`);
  console.log(`  Characters detected: ${result.characters_detected.join(', ') || 'none matched'}`);

  // Show detections per frame
  let totalFaces = 0;
  for (const frame of result.frames) {
    if (frame.faces.length > 0) {
      console.log(`    Frame @${frame.timestamp_ms}ms: ${frame.faces.length} face(s)`);
      for (const face of frame.faces) {
        const syncInfo = face.syncable
          ? `syncable (quality=${face.sync_quality.toFixed(2)})`
          : `not syncable (${face.skip_reason})`;
        console.log(`      - ${face.character_id}: bbox=${JSON.stringify(face.bbox)}, ` +
          `conf=${face.confidence.toFixed(2)}, pose=${face.head_pose.map(p => p.toFixed(1)).join(',')}, ${syncInfo}`);
      }
      totalFaces += frame.faces.length;
    }
  }
  console.log(`  Total detections: ${totalFaces}`);

  return result;
}

// Process lip-sync
async function lipsync(apiBase, videoUrl, audioUrl, requestConfig) {
  console.log('Processing lip-sync...');

  const response = await fetch(`${apiBase}/lipsync`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_url: videoUrl,
      audio_url: audioUrl,
      ...requestConfig,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API returned ${response.status}: ${errorText}`);
  }

  const result = await response.json();

  if (!result.success) {
    throw new Error(`Lip-sync failed: ${result.error_message}`);
  }

  return result;
}

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    video: null,
    audio: null,
    output: 'output.mp4',
    auto: false,
    refs: null,
    bbox: null,
    config: null,
    start: 0,
    end: null,
    fps: 3,
    api: 'http://localhost:8000',
    enhance: true,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--video':
        opts.video = args[++i];
        break;
      case '--audio':
        opts.audio = args[++i];
        break;
      case '--output':
        opts.output = args[++i];
        break;
      case '--bbox':
        opts.bbox = args[++i];
        break;
      case '--refs':
        opts.refs = args[++i];
        break;
      case '--config':
        opts.config = args[++i];
        break;
      case '--start':
        opts.start = parseInt(args[++i], 10);
        break;
      case '--end':
        opts.end = parseInt(args[++i], 10);
        break;
      case '--fps':
        opts.fps = parseInt(args[++i], 10);
        break;
      case '--api':
        opts.api = args[++i];
        break;
      case '--auto':
        opts.auto = true;
        break;
      case '--no-enhance':
        opts.enhance = false;
        break;
      case '--help':
      case '-h':
        console.log(`
Lip-Sync API Test Script

Usage:
  node test-api.mjs --video https://example.com/video.mp4 --audio https://example.com/audio.mp3 --auto
  node test-api.mjs --video https://example.com/video.mp4 --audio https://example.com/audio.mp3 --refs refs.json --auto
  node test-api.mjs --video https://example.com/video.mp4 --audio https://example.com/audio.mp3 --bbox 100,50,200,250 --end 5000

Options:
  --video      Input video URL (required)
  --audio      Audio URL for lip-sync (required)
  --output     Output video file (default: output.mp4)
  --auto       Auto-detect faces using InsightFace
  --refs       JSON file with character references for face matching
  --bbox       Face bounding box as x,y,w,h (manual, alternative to --auto)
  --config     JSON file with face configuration (for multi-face)
  --start      Start time in ms (default: 0)
  --end        End time in ms (default: video duration for --auto)
  --fps        Sample FPS for face detection (default: 3)
  --api        API base URL (default: http://localhost:8000)
  --no-enhance Disable CodeFormer enhancement

Example refs.json (for character matching):
[
  {"id": "alice", "name": "Alice", "reference_image_url": "https://example.com/alice.jpg"},
  {"id": "bob", "name": "Bob", "reference_image_url": "https://example.com/bob.jpg"}
]

Example config.json (for manual face specification):
{
  "faces": [
    {"character_id": "alice", "bbox": [100, 50, 200, 250], "start_time_ms": 0, "end_time_ms": 5000},
    {"character_id": "bob", "bbox": [400, 60, 180, 220], "start_time_ms": 1000, "end_time_ms": 5000}
  ]
}
`);
        process.exit(0);
    }
  }

  return opts;
}

// Build the request configuration for lipsync
function buildRequestConfig(opts, detectionResult = null) {
  // If config file provided, use it
  if (opts.config) {
    if (!existsSync(opts.config)) {
      throw new Error(`Config file not found: ${opts.config}`);
    }
    const config = JSON.parse(readFileSync(opts.config, 'utf-8'));
    return {
      faces: config.faces,
      enhance_quality: opts.enhance,
      fidelity_weight: config.fidelity_weight ?? 0.7,
    };
  }

  // If auto-detected faces provided
  if (detectionResult && detectionResult.frames.length > 0) {
    // Find the first frame with syncable faces
    const firstSyncableFrame = detectionResult.frames.find(f =>
      f.faces.some(face => face.syncable)
    );

    if (!firstSyncableFrame) {
      throw new Error('No syncable faces found in video');
    }

    const syncableFaces = firstSyncableFrame.faces.filter(f => f.syncable);
    const endTime = opts.end ?? detectionResult.video_duration_ms;

    return {
      faces: syncableFaces.map((face) => ({
        character_id: face.character_id,
        bbox: face.bbox,
        start_time_ms: opts.start,
        end_time_ms: endTime,
      })),
      enhance_quality: opts.enhance,
      fidelity_weight: 0.7,
    };
  }

  // Otherwise build from bbox
  if (!opts.bbox) {
    throw new Error('Either --auto, --bbox, or --config is required');
  }

  const bboxParts = opts.bbox.split(',').map(n => parseInt(n.trim(), 10));
  if (bboxParts.length !== 4) {
    throw new Error('bbox must be 4 comma-separated integers: x,y,w,h');
  }

  if (opts.end === null) {
    throw new Error('--end is required when using --bbox');
  }

  return {
    faces: [{
      character_id: 'face_1',
      bbox: bboxParts,
      start_time_ms: opts.start,
      end_time_ms: opts.end,
    }],
    enhance_quality: opts.enhance,
    fidelity_weight: 0.7,
  };
}

// Main function
async function main() {
  const opts = parseArgs();

  // Validate required options
  if (!opts.video) {
    console.error('Error: --video is required');
    process.exit(1);
  }
  if (!opts.audio) {
    console.error('Error: --audio is required');
    process.exit(1);
  }

  // Validate URLs
  if (!isUrl(opts.video)) {
    console.error('Error: --video must be a URL');
    process.exit(1);
  }
  if (!isUrl(opts.audio)) {
    console.error('Error: --audio must be a URL');
    process.exit(1);
  }

  console.log('Lip-Sync API Test');
  console.log('=================');
  console.log(`API:    ${opts.api}`);
  console.log(`Video:  ${opts.video}`);
  console.log(`Audio:  ${opts.audio}`);
  console.log(`Output: ${opts.output}`);
  console.log(`Mode:   ${opts.auto ? 'Auto-detect' : opts.config ? 'Config file' : 'Manual bbox'}`);
  console.log('');

  // Check API health first
  console.log('Checking API health...');
  try {
    const healthRes = await fetch(`${opts.api}/health`);
    const health = await healthRes.json();
    console.log(`  Status: ${health.status}`);
    console.log(`  InsightFace: ${health.insightface_loaded ? 'Loaded' : 'Not loaded'}`);
    console.log(`  MuseTalk: ${health.musetalk_loaded ? 'Loaded' : 'Not loaded'}`);
    console.log(`  LivePortrait: ${health.liveportrait_loaded ? 'Loaded' : 'Not loaded'}`);
    console.log(`  CodeFormer: ${health.codeformer_loaded ? 'Loaded' : 'Not loaded'}`);
    console.log(`  GPU: ${health.gpu_available ? `${health.gpu_name} (${health.gpu_memory_gb}GB)` : 'Not available'}`);
    console.log('');

    if (health.status !== 'healthy') {
      console.error('Warning: API is not fully healthy');
    }
  } catch (err) {
    console.error(`Error: Cannot connect to API at ${opts.api}`);
    console.error(`  ${err.message}`);
    process.exit(1);
  }

  try {
    const startTime = Date.now();

    // Auto-detect faces if requested
    let detectionResult = null;
    if (opts.auto) {
      // Load character references if provided
      let characters = [];
      if (opts.refs) {
        if (!existsSync(opts.refs)) {
          throw new Error(`Refs file not found: ${opts.refs}`);
        }
        characters = JSON.parse(readFileSync(opts.refs, 'utf-8'));
        console.log(`Loaded ${characters.length} character reference(s)`);
      } else {
        // No refs = detect all faces, auto-assign IDs
        // We still need at least one "character" to make the API happy
        // but it will auto-assign face_1, face_2, etc. for unmatched faces
        characters = [
          { id: '_detect_all', name: 'Detect All', reference_image_url: 'https://example.com/placeholder.jpg' }
        ];
        console.log('No --refs provided, will detect all faces and auto-assign IDs');
      }
      console.log('');

      detectionResult = await detectFaces(
        opts.api,
        opts.video,
        characters,
        opts.fps,
        opts.start > 0 ? opts.start : null,
        opts.end,
      );
      console.log('');
    }

    // Build request config
    const requestConfig = buildRequestConfig(opts, detectionResult);

    console.log(`Faces to process: ${requestConfig.faces.length}`);
    for (const face of requestConfig.faces) {
      console.log(`  - ${face.character_id}: bbox=${JSON.stringify(face.bbox)}, ${face.start_time_ms}ms-${face.end_time_ms}ms`);
    }
    console.log('');

    // Make request
    console.log('Sending lip-sync request...');
    const result = await lipsync(opts.api, opts.video, opts.audio, requestConfig);

    const totalTime = Date.now() - startTime;

    // Decode base64 video and save
    if (!result.output_url) {
      throw new Error('No output video in response');
    }

    const base64Data = result.output_url.replace('data:video/mp4;base64,', '');
    const videoBuffer = Buffer.from(base64Data, 'base64');
    writeFileSync(opts.output, videoBuffer);

    console.log('');
    console.log('Success!');
    console.log('========');
    console.log(`  Output: ${opts.output}`);
    console.log(`  Size: ${(videoBuffer.length / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Faces processed: ${result.faces_processed}`);
    console.log(`  Server time: ${(result.processing_time_ms / 1000).toFixed(1)}s`);
    console.log(`  Total time: ${(totalTime / 1000).toFixed(1)}s`);

    for (const faceResult of result.face_results) {
      if (faceResult.success) {
        console.log(`  - ${faceResult.character_id}: OK`);
      } else {
        console.log(`  - ${faceResult.character_id}: FAILED - ${faceResult.error_message}`);
      }
    }

  } catch (err) {
    console.error('Error:', err.message);
    process.exit(1);
  }
}

main();
