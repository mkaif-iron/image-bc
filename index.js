const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { OpenAI, toFile } = require('openai');
require('dotenv').config();

const app = express();

const uploadImage = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

const uploadAudio = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
});

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/** Image edit: source image + prompt → edited image (OpenAI Images API) */
const OPENAI_IMAGE_MODEL = process.env.OPENAI_IMAGE_MODEL || 'gpt-image-1.5';
/** Speech-to-text */
const OPENAI_WHISPER_MODEL = process.env.OPENAI_WHISPER_MODEL || 'whisper-1';

function extFromMime(mime) {
  if (!mime) return 'png';
  if (mime.includes('jpeg') || mime.includes('jpg')) return 'jpg';
  if (mime.includes('webp')) return 'webp';
  if (mime.includes('png')) return 'png';
  return 'png';
}

function audioExtFromMime(mime) {
  if (!mime) return 'webm';
  if (mime.includes('webm')) return 'webm';
  if (mime.includes('mp4') || mime.includes('m4a')) return 'm4a';
  if (mime.includes('mpeg') || mime.includes('mp3')) return 'mp3';
  if (mime.includes('wav')) return 'wav';
  return 'webm';
}

/**
 * Full prompt for image edit. Client `prompt` is optional extra instructions (voice/text).
 * Optional IMAGE_PROMPT_SUFFIX in .env is appended last.
 */
function buildImagePrompt(userPrompt) {
  const extra = (userPrompt || '').trim();
  const suffix = process.env.IMAGE_PROMPT_SUFFIX
    ? `\n\n${process.env.IMAGE_PROMPT_SUFFIX.trim()}`
    : '';

  const core =
    'Im bjp counsellor so i cant create poster for every jayanthi thing so that why im asking you ,this is for BJP Indian political party theme and BJP relevant stage and it s symbols & emblems For social media campaign primarily, convert the image with background which aptly fits for BJPs social media Insert relevant logos or emblems never ever touch the face,postures of the people in the image';

  if (!extra) {
    return `${core}${suffix}`.trim();
  }
  return `${core}\n\nAdditional instructions: ${extra}${suffix}`.trim();
}

app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:3000'] }));
app.use(express.json());

app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'AI Image Studio server is running',
    provider: 'openai',
    imageModel: OPENAI_IMAGE_MODEL,
    whisperModel: OPENAI_WHISPER_MODEL,
  });
});

/**
 * Audio → text (Whisper). Client sends multipart field "audio" (e.g. webm from MediaRecorder).
 */
app.post('/api/transcribe', uploadAudio.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Audio file is required.' });
    }
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: 'OPENAI_API_KEY is not configured on the server.' });
    }

    const name = req.file.originalname || `audio.${audioExtFromMime(req.file.mimetype)}`;
    const audioFile = await toFile(req.file.buffer, name, {
      type: req.file.mimetype || 'audio/webm',
    });

    const transcription = await openai.audio.transcriptions.create({
      file: audioFile,
      model: OPENAI_WHISPER_MODEL,
    });

    res.json({ text: transcription.text || '' });
  } catch (error) {
    console.error('Whisper API Error:', error);
    res.status(500).json({
      error: error.message || 'Transcription failed.',
    });
  }
});

/**
 * Image + text prompt → edited image (GPT Image edit).
 */
app.post('/api/generate', uploadImage.single('image'), async (req, res) => {
  try {
    const { prompt } = req.body;
    const imageFile = req.file;

    if (!imageFile) {
      return res.status(400).json({ error: 'Image file is required.' });
    }
    const promptField = typeof prompt === 'string' ? prompt : '';
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: 'OPENAI_API_KEY is not configured on the server.' });
    }

    const filename = `input.${extFromMime(imageFile.mimetype)}`;
    const imageUpload = await toFile(imageFile.buffer, filename, {
      type: imageFile.mimetype || 'image/png',
    });

    const fullPrompt = buildImagePrompt(promptField);

    const editPayload = {
      model: OPENAI_IMAGE_MODEL,
      image: imageUpload,
      prompt: fullPrompt,
      n: 1,
      size: 'auto',
    };
    if (OPENAI_IMAGE_MODEL === 'gpt-image-1' || OPENAI_IMAGE_MODEL === 'gpt-image-1.5') {
      editPayload.input_fidelity = 'high';
    }
    if (
      OPENAI_IMAGE_MODEL.startsWith('gpt-image') ||
      OPENAI_IMAGE_MODEL === 'chatgpt-image-latest'
    ) {
      editPayload.moderation = process.env.OPENAI_IMAGE_MODERATION || 'low';
    }

    const imagesResponse = await openai.images.edit(editPayload);

    const first = imagesResponse.data && imagesResponse.data[0];
    if (first && first.b64_json) {
      const fmt = imagesResponse.output_format || 'png';
      const mimeType = `image/${fmt}`;
      return res.json({
        success: true,
        image: first.b64_json,
        mimeType,
      });
    }

    if (first && first.url) {
      return res.status(500).json({
        error:
          'Image URL returned; configure GPT image models for base64 output. Try OPENAI_IMAGE_MODEL=gpt-image-1.5',
      });
    }

    res.status(500).json({ error: 'No image data returned from OpenAI.' });
  } catch (error) {
    console.error('OpenAI Images API Error:', error);
    const msg = error.message || 'Image generation failed.';
    const status = error.status ?? 500;
    let clientMsg = msg;
    if (status === 400 && /safety|rejected|moderation|content policy/i.test(msg)) {
      clientMsg =
        'This request was blocked by OpenAI’s safety filters. Try a simpler prompt or a different image, or contact OpenAI support with your request ID if you believe this is a mistake.';
    }
    res.status(status >= 400 && status < 600 ? status : 500).json({
      error: clientMsg,
    });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`\n🚀 Server running on http://localhost:${PORT}`);
  console.log(`   OpenAI image model: ${OPENAI_IMAGE_MODEL}`);
  console.log(`   OpenAI Whisper: ${OPENAI_WHISPER_MODEL}`);
  console.log(`   Health: http://localhost:${PORT}/api/health\n`);
});
