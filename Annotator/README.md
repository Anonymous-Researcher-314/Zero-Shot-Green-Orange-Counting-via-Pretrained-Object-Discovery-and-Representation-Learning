# Orange Annotator

> **A browser-based tool for annotating green oranges in 4K video frames — frame by frame, with ghost overlay tracking, auto-save, and Excel export.**

---

## What It Does

Orange Annotator is a single HTML file that runs entirely in your browser. Load a folder of extracted video frames, click on each visible green orange to place a dot, and navigate through all frames to build a complete ground truth count dataset. No installation, no server, no dependencies — just open the file and start annotating.

The tool was built specifically for annotating green (immature) oranges in 4K orchard video frames where the fruit blends into surrounding foliage, making precise dot placement essential.

---

## Quick Start

1. Download `Orange_Annotator.html`
2. Open it in any modern browser (Chrome or Edge recommended)
3. Click **Open Folder** and select your folder of extracted `.jpg` frames
4. Set **Tree ID** and **Video ID** in the config bar (e.g. `Tree_01`, `Vid 01`)
5. Click on each visible green orange to place a numbered dot
6. Press `→` or `D` to go to the next frame
7. When all frames are done, click **Export XLSX** to save your ground truth

---

## The Ghost Overlay — Never Lose Your Place

The most important feature of the annotator is the **Ghost Overlay**.

When you annotate a frame and move to the next one, the dot positions from the **previous frame** are shown as faint translucent markers on the current frame. These ghost dots are not real annotations — they are a visual reference only.

**Why this matters:** Green oranges do not move between consecutive frames. The ghost overlay shows you exactly where you marked oranges in the last frame, so you can:

- Immediately see which oranges are in roughly the same position
- Spot oranges that have moved slightly or become newly visible
- Avoid re-scanning the entire frame from scratch every time
- Never lose count by forgetting which regions you already covered

Without the ghost overlay, annotating dense canopy frames requires re-scanning the entire image on every frame — which is slow and error-prone. With it, you only need to check what changed.

The ghost overlay is **enabled by default**. You can toggle it with the checkbox in the config bar, and change its colour (magenta, cyan, yellow, white) to maximise contrast against different frame backgrounds.


---

## Features

### Annotation
- **Click to place dot** — click anywhere on the canvas to add a numbered dot at that position
- **Auto-numbering** — dots are numbered sequentially starting from 1
- **Undo last dot** — `Ctrl+Z` removes the most recently placed dot
- **Remove by Alt+Click** — hold `Alt` and click near any dot to remove the nearest one; remaining dots are automatically renumbered
- **Remove by number** — type a number in the Remove # box and press Enter to delete a specific dot by its number; all subsequent dots renumber automatically
- **Clear frame** — remove all dots from the current frame with `C` or the Clear button

### Ghost Overlay
- **Previous frame tracking** — the dot positions from frame N−1 are shown as faint ghost markers on frame N
- **Colour selection** — choose ghost colour (magenta, cyan, yellow, white) to contrast with your specific frames
- **Toggle on/off** — checkbox in the config bar; change takes effect immediately
- **Visual reference only** — ghost dots are never saved or exported

### Navigation
- **Keyboard navigation** — `←` / `→` or `A` / `D` to move between frames
- **Sidebar click** — click any frame in the left sidebar to jump directly to it
- **Frame counter** — top right shows current frame number and total
- **Prefetch** — next and previous frames are preloaded in the background for instant navigation

### Zoom and Pan
- **Zoom** — `Ctrl + Scroll` to zoom in/out; zoom level shown in bottom right
- **Pan** — `Scroll` to pan up/down; `Shift + Scroll` to pan left/right
- **Drag pan** — hold `Shift` and drag, or use middle mouse button drag
- **Fit to screen** — press `F` to fit the current frame to the canvas

### Sidebar
- Frames listed in order with their dot count
- Frames with at least one dot shown in green with a count badge
- Unannotated frames shown in grey with `—`
- Progress bar at the bottom shows percentage of frames annotated
- Per-frame `+` / `−` buttons to manually adjust counts without editing dots

### Auto-Save and Session Restore
- Annotations are **automatically saved to browser localStorage** every 800ms after any change
- Status indicator in the bottom bar shows `● saved` / `● saving...`
- If you close the browser and reopen the annotator, a **session banner** appears offering to restore your previous session
- Session is keyed by the folder name so multiple annotation sessions are kept separately
- **Save Progress** button saves a JSON progress file to disk containing all annotations, frame names, tree ID, video ID, and a timestamp — this is the primary durable backup

### Import and Export
- **Export XLSX** — exports a `.xlsx` file with one row per frame: `image_filename`, `tree_id`, `video_id`, `frame_id`, `ground_truth_count`
- **Export Labeled Images** — exports annotated frames as `.png` files with dots drawn on them (annotated only, or all frames)
- **Save Frame** — press `S` or the Save Frame button to download the current frame with its dots as a PNG
- **Import XLSX** — load a previously exported Excel file to restore counts (useful for cross-checking or resuming from an Excel backup)
- **Save Progress** — saves a JSON file with full dot coordinates, frame names, tree/video IDs, and timestamp

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` or `D` | Next frame |
| `←` or `A` | Previous frame |
| `Ctrl + Z` | Undo last dot |
| `Alt + Click` | Remove nearest dot (auto-renumbers) |
| `C` | Clear all dots on current frame |
| `F` | Fit image to screen |
| `S` | Save current frame as PNG with dots |
| `Ctrl + Scroll` | Zoom in / out |
| `Scroll` | Pan up / down |
| `Shift + Scroll` | Pan left / right |
| `Shift + Drag` | Pan canvas |
| `Middle mouse drag` | Pan canvas |

---

## Config Bar

| Field | Description |
|-------|-------------|
| **Tree ID** | The tree identifier — auto-detected from filename if possible (e.g. `Tree_01`) |
| **Video ID** | The video identifier — auto-detected from filename if possible (e.g. `Vid 01`) |
| **Ghost overlay** | Toggle the previous-frame dot overlay on/off |
| **Ghost colour** | Colour of the ghost dots (magenta / cyan / yellow / white) |

Tree ID and Video ID are used in the exported XLSX and saved JSON files. They are auto-detected from the filename pattern `{TreeID}_Vid {N}_F{frame}.jpg` when you load a folder.

---

## Annotation Protocol

For consistent results across multiple sessions and annotators:

1. **Count only green and transitioning oranges** — do not count flower buds, early bud formations, or fully ripe (orange-coloured) fruit
2. **Count if at least 50% visible** — annotate an orange if at least half of its visible perimeter is within the frame, regardless of occlusion by leaves, branches, or other fruit
3. **One dot per orange** — place the dot approximately at the centre of the visible fruit
4. **Use the ghost overlay** — always keep ghost overlay enabled; it shows where you placed dots in the previous frame so you can track oranges as they move slightly between frames and avoid missing any

---

## Saved File Formats

### JSON Progress File (Save Progress button)

```json
{
  "treeId": "Tree_01",
  "videoId": "Vid 01",
  "totalFrames": 301,
  "frameNames": ["Tree_01_Vid 01_F001.jpg", "Tree_01_Vid 01_F002.jpg", "..."],
  "annotations": {
    "0": [{"x": 219.1, "y": 1069.3, "manual": true}, {"x": 958.8, "y": 2192.9, "manual": true}],
    "1": [{"x": 221.4, "y": 1071.0, "manual": true}],
    "...": []
  },
  "currentFrame": 150,
  "savedAt": "2026-05-15T05:53:04.328Z"
}
```

- `frameNames` — ordered list of all frame filenames
- `annotations` — dictionary keyed by frame index (string), value is array of dot objects with pixel coordinates `x`, `y` in the native 3840×2160 frame and a `manual` boolean flag
- `currentFrame` — last frame visited (for resuming)
- `savedAt` — ISO 8601 timestamp of the last save

### Excel Export (Export XLSX button)

One row per frame, columns:

| Column | Description |
|--------|-------------|
| `image_filename` | Full filename of the frame (e.g. `Tree_01_Vid 01_F001.jpg`) |
| `tree_id` | Tree identifier from config bar |
| `video_id` | Video identifier from config bar |
| `frame_id` | Sequential frame number |
| `ground_truth_count` | Number of dots placed on this frame |

---

## Loading Frames

### Open Folder (recommended)
Click **Open Folder** and select the folder containing your extracted `.jpg` frames. The annotator loads all image files in the folder, sorts them by filename, and detects Tree ID and Video ID automatically from the filename pattern.

### Open Files
Click **Open Files** to select individual image files if you do not want to load a whole folder.

### Supported Filename Pattern
The tool auto-detects Tree ID and Video ID from filenames matching:
```
{TreeID}_Vid {N}_F{frameNum}.jpg
```
For example: `Tree_01_Vid 01_F001.jpg`

---

## Browser Requirements

- Chrome or Edge recommended (required for the File System Access API used by Open Folder)
- Firefox supported via Open Files (folder selection may not work)
- No internet connection required after the page loads (fonts load from Google Fonts on first open)
- No installation, no server, no Python, no Node.js

---

## Tips

- **Annotate in one session per video** — the auto-save and ghost overlay work best when you annotate one video from start to finish without switching between videos
- **Save Progress frequently** — the JSON file is your durable backup; localStorage can be cleared by the browser
- **Check the progress bar** — the sidebar footer shows what percentage of frames have been annotated; aim for 100% before exporting
- **Use zoom for dense frames** — `Ctrl+Scroll` to zoom into dense canopy areas where oranges are small or partially occluded
- **Ghost colour contrast** — if your frames have a lot of pink/magenta tones, switch the ghost colour to cyan or yellow for better visibility