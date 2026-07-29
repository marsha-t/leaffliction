# leaffliction

## Setup

Create and activate a virtual environment.

```bash
python3 -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

## Usage

Run the dataset analysis on a plant directory.

```bash
./Distribution.py data/Apple
./Distribution.py ./data/Apple
```

Apply augmentations on an image

```bash
./Augmentation.py 'data/Apple/Apple_Black_rot/image (1).JPG'
```