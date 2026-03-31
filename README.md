# 👁️‍🧠 Face Recognition Attendance System

A Machine Learning–powered smart attendance system that uses **face encoding + background context** for highly accurate identity recognition. Built with **FastAPI**, **SFace (ONNX)**, and **MediaPipe**.

> 🔗 **Live API:** [https://face-recogination.onrender.com](https://face-recogination.onrender.com)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧬 **Face Encoding** | SFace (ONNX) deep facial feature extraction for robust identity matching |
| 🎨 **Background Context** | MediaPipe-based background color encoding for enhanced accuracy |
| 📸 **Real-time Recognition** | Live camera feed → Base64 → instant attendance marking |
| ⏰ **Auto-Finalize** | GitHub Action auto-marks OUT at midnight for users who forgot |
| 📊 **Attendance Logs** | Full history with in/out times, dates, and status tracking |
| ☁️ **Cloud Storage** | User images stored via Cloudinary, encodings in NeonDB (PostgreSQL) |

**Accuracy:** 90%+

---

## 🏗️ System Architecture

The system consists of **3 key layers**:

- **Frontend (Kotlin App)** — Captures live user images via camera, converts to Base64, and communicates with backend APIs for registration and recognition.
- **Django Backend** — Manages user data, handles Cloudinary image uploads, and routes registration requests to the ML service.
- **FastAPI ML Service** — Performs SFace face encoding + MediaPipe background encoding, compares against stored encodings, and marks attendance in real time.
- **NeonDB (PostgreSQL)** — Stores user face encodings, background data, and attendance logs.
- **Cloudinary (CDN)** — Stores user registration images.
- **GitHub Actions** — Runs a daily cron job to auto-finalize incomplete attendance records.

### 📲 Flow 1: Registration (Django → ML)
Kotlin app sends user details to Django → Django stores the user and uploads image to **Cloudinary** → FastAPI fetches the image, performs **SFace face encoding** and **MediaPipe background encoding**, and saves the encodings to the database.

### 📷 Flow 2: Recognition (Kotlin App → ML)
Kotlin app captures live image → converts to **Base64** → sends to FastAPI → FastAPI decodes the image, encodes face + background, compares with stored encodings → if similarity is high, attendance is marked as **present** with timestamp.

### 📊 Flow 3: Attendance Log (Database → Kotlin App)
Kotlin app requests attendance history for a given `user_id` → FastAPI fetches records from **NeonDB** → returns a list of attendance logs sorted by date with `in_time`, `out_time`, and status.

### 🔄 Flow 4: Auto-Finalize (GitHub Action)
Runs daily at **12:01 AM IST** → finds users who checked IN but forgot to check OUT → auto-sets `out_time` to 11:59 PM and marks status as `"present"`.

---

## 📂 Project Structure

```
Face_Recognition_Attendance_System/
├── app.py                          # FastAPI server & API routes
├── config.py                       # Cloudinary configuration
├── requirements.txt                # Python dependencies
│
├── core/                           # ML Logic
│   ├── register_face.py            # Face registration (SFace + MediaPipe encoding)
│   ├── recogination_face.py        # Face recognition & attendance marking
│   └── shared_code.py              # Shared utilities (encoding, comparison)
│
├── DB/                             # Database Layer
│   ├── database.py                 # Async SQLAlchemy engine & ORM models
│   ├── create_table.py             # Table creation script
│   └── init_db.py                  # Database initialization
│
├── scripts/                        # Automation Scripts
│   ├── auto_finalize_attendance.py # Auto-finalize incomplete attendance
│   ├── run_finalize.py             # Entry point for finalize script
│   ├── reset_db.py                 # Database reset utility
│   ├── base64_convert.py           # Image to Base64 conversion
│   └── download_sface.py           # Download SFace ONNX model
│
├── models/                         # ML Models
│   └── face_recognition_sface_2021dec.onnx
│
├── tests/                          # Test Suite
│   ├── db_test.py                  # Database connectivity tests
│   └── image_test.py               # Image processing tests
│
└── .github/workflows/
    └── auto_finalize.yml           # Daily attendance finalizer (GitHub Action)
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+

### Setup

```bash
# Clone the repository
git clone https://github.com/Sparsh2509/Face_Recognition_Attendance_System.git
cd Face_Recognition_Attendance_System

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Cloudinary
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# NeonDB (PostgreSQL)
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name

PYTHONPATH=.
```

---

## 🚦 Running the API

```bash
uvicorn app:app --reload
```

| URL | Description |
|---|---|
| [http://127.0.0.1:8000](http://127.0.0.1:8000) | Root endpoint |
| [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Swagger API Docs |
| [https://face-recogination.onrender.com](https://face-recogination.onrender.com) | Production (Render) |

---

## 📡 API Endpoints

### 1. `POST /register/` — Register a User

Registers a new user by downloading their image from Cloudinary, encoding face + background, and saving to the database.

**Request:**
```json
{
  "user_id": "3",
  "name": "Sparsh Gupta",
  "image_url": "https://res.cloudinary.com/demo/image/upload/v1720000000/user3.jpg"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Sparsh Gupta registered successfully.",
  "user_id": "3",
  "name": "Sparsh Gupta"
}
```

---

### 2. `POST /recognize/` — Recognize & Mark Attendance

Receives a live camera frame as Base64, compares against stored encodings, and marks attendance.

**Request:**
```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAYGBg...",
  "mode": "in"
}
```

> `mode` can be `"in"` (check-in) or `"out"` (check-out)

**Response:**
```json
{
  "status": "success",
  "message": "Attendance marked",
  "can_retry": false,
  "data": {
    "status": "present",
    "mode": "in",
    "time": "16:51",
    "date": "2025-10-21",
    "user_id": "3",
    "name": "Sparsh Gupta"
  }
}
```

---

### 3. `GET /attendance-log/?user_id={id}` — Fetch Attendance History

Returns all attendance records for a user, sorted by date (newest first).

**Request:**
```
GET /attendance-log/?user_id=3
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "user_id": "3",
      "name": "Sparsh Gupta",
      "date": "2025-10-21",
      "in_time": "16:51:08.439476",
      "out_time": null,
      "in_status": "present",
      "out_status": "absent"
    },
    {
      "user_id": "3",
      "name": "Sparsh Gupta",
      "date": "2025-07-01",
      "in_time": "09:56:43.944331",
      "out_time": "09:56:55.737586",
      "in_status": "present",
      "out_status": "present"
    }
  ]
}
```

---

## ⏰ Auto-Finalize Attendance (GitHub Action)

A scheduled GitHub Action runs **daily at 12:01 AM IST** to automatically finalize incomplete attendance records.

**What it does:**
- Finds all users who checked **IN** but forgot to check **OUT** yesterday
- Auto-sets their `out_time` to `11:59 PM` and marks `out_status` as `"present"`

**Configuration:**
- Workflow file: [`.github/workflows/auto_finalize.yml`](.github/workflows/auto_finalize.yml)
- Schedule: `cron: '31 18 * * *'` (18:31 UTC = 12:01 AM IST)
- Can also be triggered manually from the **Actions** tab

**Required GitHub Secrets:**
| Secret | Description |
|---|---|
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password |
| `DB_HOST` | Database host URL |
| `DB_PORT` | Database port (default: 5432) |
| `DB_NAME` | Database name |

---

## 🧪 Testing

```bash
# Test database connectivity
python tests/db_test.py

# Test image processing
python tests/image_test.py
```

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| **FastAPI** | REST API server |
| **SFace (ONNX)** | Face feature extraction |
| **MediaPipe** | Background color encoding |
| **SQLAlchemy (Async)** | ORM & database operations |
| **NeonDB (PostgreSQL)** | Cloud database |
| **Cloudinary** | Image storage CDN |
| **GitHub Actions** | Automated attendance finalization |
| **Render** | API deployment |

---
