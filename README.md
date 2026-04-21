# Steepest Descent Optimizer — Full Stack App

Based on: Napitupulu et al. (2018), IOP Conf. Ser.: Mater. Sci. Eng. 332 012024

## Folder Structure

```
steepest-descent/
├── backend/
│   ├── main.py          ← FastAPI app (all endpoints)
│   ├── optimizer.py     ← Core math: all step size methods
│   └── requirements.txt
├── frontend/
│   ├── index.html       ← Self-contained React app (Vite-free)
│   └── (or Next.js if you prefer)
└── README.md
```

## Quick Start — Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

## Backend API

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | /functions | List all test functions + metadata |
| POST   | /optimize  | Run one method, return full path   |
| POST   | /compare   | Run all methods side-by-side       |
| POST   | /contour   | Compute Z grid for contour plot    |

Example request:
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"func_name":"booth","method":"BB1","x0":[-4,4],"max_iter":1000}'
```

## Step Size Methods

| Key  | Name                | Reference              |
|------|---------------------|------------------------|
| C    | Cauchy exact        | Cauchy (1847)          |
| A    | Armijo              | Armijo (1966)          |
| B    | Backtracking        | Dennis & Schnabel      |
| BB1  | Barzilai-Borwein 1  | Barzilai & Borwein (1988)|
| BB2  | Barzilai-Borwein 2  | Barzilai & Borwein (1988)|
| EL   | Elimination         | Wen et al. (2012)      |

## Deploy (free tier)

- Backend → Render.com: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Frontend → Vercel: deploy the `frontend/` folder as static HTML
