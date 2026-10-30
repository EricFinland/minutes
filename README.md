# Minutes

A small meeting transcriber for Windows and macOS that runs entirely on your own computer.

It listens to two things at once: your microphone ("Me") and whatever your computer is
playing ("Them", meaning everyone else on the Zoom, Meet or Teams call). It writes a live
transcript as people talk, saves it to disk line by line, and can turn a meeting into a
short summary with a local Ollama model when you ask.

## Starting it

Double-click `run.pyw`.

Or from a terminal in this folder:

```
C:\Python314\pythonw.exe run.pyw
```

The first launch downloads the speech model (about 1.6 GB, once). The status line says
"Loading speech model..." until it is ready. After that, starting takes a few seconds.

If it is already running, launching it again just brings the existing window back.

## On a Mac

Needs macOS 13 (Ventura) or newer and Python 3.10+ (`brew install python@3.12`).

1. In Terminal, from this folder, run `bash mac/install.sh`. It sets up a private Python
   environment and downloads the speech model (about 1.6 GB, once).
2. Double-click `mac/Minutes.command` to start Minutes. The Terminal window can be closed.
3. The first time, macOS asks for **Microphone** and **Screen & System Audio Recording**
   access. Allow both, quit Minutes (... menu > Quit) and start it again. The call audio
   ("Them") comes from screen recording permission; Minutes only keeps the audio, never
   the picture.

Differences from Windows:

- The hotkey is **Ctrl+Option+M**.
- "Start with Windows" is called **Open at login**.
- Minutes has no Dock icon and is not in Cmd-Tab.
- Hiding from screen share works for screenshots, QuickTime and most apps, but on macOS 15
  and newer some screen sharing tools ignore it. Check with a test share before relying
  on it.
- Prefer BlackHole for call audio? Set `speaker_device` to its name (for example
  `"BlackHole 2ch"`) in settings.json and Minutes records it as the "Them" side.

## Using it

Minutes starts recording right away and shows a small pill near the edge of the screen.
Press **Ctrl+Alt+M** at any time to show or hide it.

On the pill:

- **Red dot**: recording. Grey means paused.
- **Timer**: how long the current meeting has been going.
- **Me / Them**: lights up while that side is talking.
- **Record / stop button**: pause or resume recording.
- **Expand button**: open the panel with the live transcript.

In the panel:

- **Copy**: copy the whole transcript to the clipboard.
- **Open**: open `transcript.txt` in your text editor.
- **Summarize**: write a summary (overview, key points, decisions, action items, open
  questions) and open it when done. Takes a minute or two, a few minutes for a long
  meeting.
- **New meeting**: finish this meeting and start a fresh one.
- **...** menu:
  - **Open meetings folder**: all your meetings in Explorer.
  - **Previous meetings**: jump to an earlier meeting.
  - **Polish transcript**: re-listen to the saved audio with more context for a cleaner
    transcript (`transcript_polished.txt`). Slower than real time is normal. Only works
    while the meeting's audio still exists.
  - **Start with Windows**: launch Minutes when you sign in.
  - **Quit Minutes**: stop recording, save everything and exit.

Closing the window only hides it. Recording keeps going.

If nobody says anything for 15 minutes, the meeting is closed automatically and the next
time someone talks a new meeting begins.

## Where your files go

```
C:\Users\ericc\Minutes\
  settings.json
  meetings\
    2026-10-29_140312\
      transcript.txt            the live transcript
      lines.jsonl               the same, one line per entry (used by the app)
      summary.md                after you click Summarize
      transcript_polished.txt   after you run Polish transcript
      audio_me.wav              your microphone
      audio_them.wav            the call audio
      meta.json
```

Each line in the transcript looks like `[00:12:31] Them: Let's push the launch a week.`

## Privacy

- Everything runs on this computer. Speech recognition is local, and summaries use Ollama
  on `localhost`. Nothing is uploaded anywhere.
- The Minutes window is invisible to screen sharing and screen recording, and it does not
  appear in the taskbar or Alt-Tab.
- Audio recordings are deleted automatically after 7 days. Transcripts and summaries are
  kept until you delete them.

## Settings

Edit `C:\Users\ericc\Minutes\settings.json` (on a Mac, `~/Minutes/settings.json`; create it
if it is missing) and restart
Minutes. Only add the keys you want to change, for example:

```json
{
  "language": "en",
  "keep_audio_days": 7,
  "meeting_gap_minutes": 15,
  "record_on_launch": true,
  "hotkey_toggle": "ctrl+alt+m",
  "summary_model": "llama3:latest"
}
```

| Key | What it does |
|---|---|
| `language` | Spoken language, e.g. `"en"`, `"es"`, `"de"`. |
| `whisper_model` | Speech model. `"large-v3-turbo"` (default, best). `"distil-large-v3"` is faster but English only. |
| `cpu_threads` | CPU threads for transcription (default: 8, or fewer on machines with fewer cores). |
| `keep_audio_days` | Days to keep audio WAVs before deleting them. |
| `meeting_gap_minutes` | Minutes of silence before a meeting is closed automatically. |
| `record_on_launch` | Start recording as soon as Minutes opens. |
| `hotkey_toggle` | Show/hide shortcut, e.g. `"ctrl+alt+m"`. |
| `mic_device`, `speaker_device` | Use a specific microphone or output device by name instead of the Windows default. |
| `summary_model` | Ollama model used by Summarize. |
| `vad_threshold` | Speech detection sensitivity (lower picks up quieter speech). |

## Troubleshooting

**Nothing shows up for "Them".** Minutes records whatever plays through the Windows
default output device. Check that the call audio is going to that device (Settings >
System > Sound > Output). If you switch to headphones mid-call, give it a few seconds to
follow the new device.

**Nothing shows up for "Me".** Check the default input device in Windows sound settings
and that Python is allowed to use the microphone (Settings > Privacy & security >
Microphone).

**Mac: nothing shows up for "Them", or a permission error.** Open System Settings >
Privacy & Security > Screen & System Audio Recording and make sure Terminal (or Python,
if you use Open at login) is allowed, then restart Minutes. Do the same under Microphone
if "Me" is empty.

**Summarize fails.** Summaries need Ollama. Minutes tries to start it for you; if that
does not work, open the Ollama app (or run `ollama serve`) and try again. If it says the
model is missing, run the `ollama pull` command it shows, for example:

```
ollama pull llama3:latest
```

**Polish says the audio is no longer available.** The recording was older than
`keep_audio_days` and was deleted. The original transcript is still there.

**Lines appear twice when using speakers.** Minutes removes most echo automatically, but
headphones give the cleanest "Me" transcript.
