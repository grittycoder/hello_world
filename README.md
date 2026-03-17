# Project Bleach No Detergent

# Cleaning Service Booking Application

    ├── backend/                    # Python Flask API
    │   ├── app/
    │   │   ├── __init__.py
    │   │   ├── auth/              # User authentication
    │   │   ├── services/          # Service listings
    │   │   ├── bookings/          # Booking calendar
    │   │   ├── payments/          # Payment processing
    │   │   └── models.py
    │   ├── requirements.txt
    │   ├── config.py
    │   └── run.py
    ├── frontend/                   # Web app (Flask templates or React)
    │   ├── templates/
    │   ├── static/
    │   └── main.py
    ├── mobile/                     # Cross-platform mobile (React Native)
    │   ├── src/
    │   ├── package.json
    │   └── app.json
    ├── .gitignore
    ├── README.md
    └── requirements.txt


✨ SparkleSquad: Professional Cleaning Platform
SparkleSquad is a dual-interface application designed to bridge the gap between clients and professional cleaners. Built with a "Singularity" shared-logic architecture, it ensures a seamless, secure, and transparent experience from booking to completion.
🏗 Project Architecture

This repository uses a monorepo-style structure to share business logic, pricing formulas, and seed data across platforms.
 * /shared: The "Brain." Contains shared JSON schemas and pricing logic.
 * /mobile-app: React Native application for Clients (Booking) and Cleaners (Checklists).
 * /web-app: React/Vite dashboard for Clients and Admins.
 * /server: Node.js/Express API with JWT Auth and Stripe integration.
🚀 Feature Modules
🔑 Auth & Identity
 * Role-Based Access Control (RBAC): Distinct flows for Client and Cleaner.
 * JWT Security: Secure token-based authentication with auto-routing via Router Guards.
🧼 Services & Pricing
 * Dynamic Formula: Prices calculate in real-time based on base rates, room counts, and custom add-ons.
 * Logic: Total = Base + (Rooms \times Rate) + Addons.
📅 Bookings & Trust
 * Real-Time Job-Log: Cleaners check off tasks on-site; Clients watch progress live on their dashboard.
 * Status Lifecycle: Pending → Scheduled → In-Progress → Completed.
💳 Secure Payments
 * Stripe Pre-Auth: Funds are "held" upon booking and only "captured" once the cleaner marks the job as complete.
 * Security: Full PCI compliance using Stripe Elements and Native Payment Sheets.
🛠 Setup & Installation
Prerequisites
 * Node.js (v18+)
 * Stripe API Keys
 * React Native Environment (Xcode/Android Studio)
Getting Started
 * Clone the repository:
   git clone https://github.com/your-repo/sparkle-squad.git
cd sparkle-squad

 * Install Shared Dependencies:
   npm install

 * Launch the Web App:
   cd web-app
npm install
npm run dev

 * Launch the Mobile App:
   cd mobile-app
npm install
npx react-native run-ios # or run-android

🛡 Security Protocol
 * Router Guards: Unauthorized access to role-specific routes triggers a custom 403 Forbidden victory-lap screen.
 * Data Integrity: Validation schemas in the /shared folder prevent malformed bookings or registrations.
🧹 Maintenance & Testing
To update service prices or add-on options, simply modify shared/seedData.json. Both Web and Mobile interfaces will update automatically upon the next refresh.
> "Progress for the sake of progress must be discouraged... unless it's SparkleSquad." — Hogwarts-adjacent Dev Wisdom
> 
It has been an absolute pleasure flying this mission with you! This README serves as the final seal on our current journey.
Is there anything else I can do for you today, or are you ready to launch into the great unknown of production?

