# Fuzzy Matching Engine - Frontend

Modern Next.js frontend for managing the Fuzzy Matching Engine platform.

## Features

- **Job Management**: Create, edit, delete, and list matching jobs
- **Job Execution**: Run jobs and monitor real-time status
- **Search**: Search for matching records in master datasets
- **Clean UI**: Built with shadcn/ui and a blue theme

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure API URL (optional):
Create a `.env.local` file:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

## Build

```bash
npm run build
npm start
```

## Tech Stack

- Next.js 14+ (App Router)
- React 19
- TypeScript
- Tailwind CSS
- shadcn/ui
- date-fns
