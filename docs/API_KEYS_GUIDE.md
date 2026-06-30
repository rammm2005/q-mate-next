# API Keys Configuration Guide

## 📋 Required API Keys

### 1. Google Gemini API Key (REQUIRED) ⭐

**Purpose:** 
- Generate AI-powered answers
- Process and classify queries
- Evaluate retrieval results

**How to Get:**
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key

**Free Tier Limits:**
- 60 requests per minute
- 1,500 requests per day
- Very generous for development

**Configuration:**
```env
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**Testing:**
```bash
# Test if key works
curl https://generativelanguage.googleapis.com/v1/models?key=YOUR_KEY
```

---

## 🔧 Optional API Keys

### 2. Supabase (OPTIONAL - For Production)

**Purpose:**
- Persistent vector storage
- PostgreSQL with pgvector
- Production-grade database

**When Needed:**
- Production deployment
- Multi-user environments
- Persistent data storage

**How to Get:**
1. Visit: https://supabase.com/dashboard
2. Create a new project
3. Go to **Settings → API**
4. Copy **URL** and **anon/public key**

**Configuration:**
```env
SUPABASE_URL=https://xxxxxxxxxxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Note:** Current demo uses in-memory storage. Database not required for local development.

---

### 3. GitHub Personal Access Token (OPTIONAL)

**Purpose:**
- Clone private repositories
- Higher rate limits (5,000 requests/hour vs 60/hour)
- Access organization repositories

**When Needed:**
- Indexing private repositories
- Heavy GitHub API usage
- Organization repositories

**How to Get:**
1. Visit: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Select scopes:
   - ✅ `repo` (for private repos)
   - ✅ `public_repo` (for public repos only)
4. Generate and copy token

**Configuration:**
```env
GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**Rate Limits:**
- Without token: 60 requests/hour
- With token: 5,000 requests/hour

**Security Note:** Never commit tokens to version control!

---

### 4. Hugging Face Token (OPTIONAL)

**Purpose:**
- Faster model downloads
- Access gated models
- Avoid download throttling

**When Needed:**
- First-time IndoBERT download
- Gated model access
- Faster model loading

**How to Get:**
1. Visit: https://huggingface.co/settings/tokens
2. Create account if needed
3. Click **"New token"**
4. Select **"Read"** access
5. Generate and copy token

**Configuration:**
```env
HUGGING_FACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**Benefits:**
- Faster downloads (no throttling)
- Access to gated models
- Higher rate limits

---

## 🎛️ Configuration Parameters (Optional)

### Application URLs

```env
# Backend API URL
BACKEND_URL=http://localhost:8000

# Frontend URL
FRONTEND_URL=http://localhost:3000
```

### Model Configuration

```env
# IndoBERT model name
INDOBERT_MODEL=firqaaa/indo-sentence-bert-base

# Alternative: English-only model
# INDOBERT_MODEL=all-MiniLM-L6-v2
```

### BM25 Parameters

```env
# Term frequency saturation (1.5 - 2.0)
# Higher = more weight on term frequency
BM25_K1=1.8

# Length normalization (0.0 - 1.0)
# 0 = no normalization, 1 = full normalization
BM25_B=0.75
```

### Hybrid Retrieval

```env
# Alpha weight for BM25 vs Semantic (0.0 - 1.0)
# 0.0 = 100% semantic, 1.0 = 100% BM25
HYBRID_ALPHA=0.5
```

### Similarity Threshold

```env
# Minimum cosine similarity (0.0 - 1.0)
# Higher = more precise, lower recall
SIMILARITY_THRESHOLD=0.15
```

### Performance

```env
# Maximum chunk size in tokens
MAX_CHUNK_SIZE=512

# Token overlap between chunks
CHUNK_OVERLAP=50

# Number of results to return
TOP_K_RESULTS=20
```

### Logging

```env
# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

---

## 🚀 Quick Setup

### Minimal Setup (Development)

```env
# Only this is required for local development:
GEMINI_API_KEY=your_key_here
```

### Recommended Setup (Development)

```env
# Required
GEMINI_API_KEY=your_key_here

# Recommended for better experience
GITHUB_TOKEN=your_token_here
HUGGING_FACE_TOKEN=your_token_here
```

### Production Setup

```env
# Required
GEMINI_API_KEY=your_key_here

# Production database
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Optional but recommended
GITHUB_TOKEN=your_token_here
LOG_LEVEL=WARNING
NODE_ENV=production
```

---

## 📝 Step-by-Step Setup

### Step 1: Get Gemini API Key

1. Open: https://aistudio.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key

### Step 2: Create .env.local

```bash
# In project root directory
cp .env.local.example .env.local
```

### Step 3: Edit .env.local

```bash
# Windows
notepad .env.local

# Mac/Linux
nano .env.local
```

### Step 4: Add Your Key

```env
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Step 5: Save and Test

```bash
# Start backend
cd backend
python -m uvicorn app.main:app --reload

# Start frontend (new terminal)
cd frontend
npm run dev

# Open browser
http://localhost:3000
```

---

## 🔒 Security Best Practices

### 1. Never Commit API Keys

```bash
# .gitignore should include:
.env.local
.env
*.env
```

### 2. Use Environment Variables

```bash
# Set temporarily (Linux/Mac)
export GEMINI_API_KEY=your_key

# Set temporarily (Windows)
set GEMINI_API_KEY=your_key
```

### 3. Rotate Keys Regularly

- Rotate API keys every 90 days
- Immediately rotate if exposed
- Use different keys for dev/prod

### 4. Limit Key Permissions

- Use read-only tokens when possible
- Restrict GitHub token scopes
- Use separate keys per environment

### 5. Monitor Usage

- Check Gemini usage: https://aistudio.google.com
- Check GitHub rate limits: `curl -H "Authorization: token TOKEN" https://api.github.com/rate_limit`
- Set up billing alerts

---

## 🐛 Troubleshooting

### Error: "GEMINI_API_KEY not found"

**Solution:**
1. Check `.env.local` exists in root directory
2. Verify key name is exactly `GEMINI_API_KEY`
3. No spaces around `=`
4. Restart backend after changes

```bash
# Correct
GEMINI_API_KEY=AIzaSy...

# Wrong
GEMINI_API_KEY = AIzaSy...  # Space around =
GEMINI_KEY=AIzaSy...         # Wrong name
```

### Error: "Invalid API key"

**Solution:**
1. Verify key at: https://aistudio.google.com/app/apikey
2. Check for copy/paste errors
3. Ensure no extra spaces or newlines
4. Try generating a new key

### Error: "Rate limit exceeded"

**Solution:**
```
Free tier: 60 requests/minute
Wait 1 minute before retrying
Or upgrade to paid plan
```

### Error: "Model not found" (IndoBERT)

**Solution:**
```bash
# First run downloads ~500MB model
# Be patient, takes 2-5 minutes

# Check download progress:
tail -f backend.log

# If download fails, set Hugging Face token:
HUGGING_FACE_TOKEN=your_token
```

### Error: "GitHub rate limit"

**Without token:** 60 requests/hour
**With token:** 5,000 requests/hour

**Solution:** Add GitHub token to `.env.local`

---

## 📊 API Key Comparison

| Service | Required? | Free Tier | Use Case |
|---------|-----------|-----------|----------|
| **Gemini** | ✅ Yes | 60 req/min | Answer generation |
| **Supabase** | ❌ No | 500MB DB | Production database |
| **GitHub** | ❌ No | 60 req/hour | Private repos |
| **Hugging Face** | ❌ No | Unlimited | Model downloads |

---

## 💰 Cost Estimation

### Free Tier Usage (Development)

**Gemini Free Tier:**
- 60 requests/minute
- 1,500 requests/day
- **Cost: $0/month**
- Sufficient for: Development, testing, small demos

**Estimated Queries:**
- 10 queries/hour = **720 queries/day** (well within limits)
- Average query: 2-3 Gemini API calls

### Paid Usage (Production)

**Gemini Pricing:**
- $0.00025 per 1K characters input
- $0.0005 per 1K characters output
- **Estimated: $5-10/month** for 10K queries

**Supabase Pricing:**
- Free: 500MB DB, 2GB bandwidth
- Pro: $25/month (8GB DB, 100GB bandwidth)

**Total Estimated Cost:**
- Development: **$0/month**
- Small production: **$5-10/month**
- Medium production: **$30-50/month**

---

## 🔗 Useful Links

### Get API Keys
- **Gemini**: https://aistudio.google.com/app/apikey
- **Supabase**: https://supabase.com/dashboard
- **GitHub**: https://github.com/settings/tokens
- **Hugging Face**: https://huggingface.co/settings/tokens

### Documentation
- **Gemini API Docs**: https://ai.google.dev/docs
- **Supabase Docs**: https://supabase.com/docs
- **GitHub API Docs**: https://docs.github.com/en/rest
- **Hugging Face Docs**: https://huggingface.co/docs

### Monitoring
- **Gemini Usage**: https://aistudio.google.com
- **GitHub Rate Limit**: `curl https://api.github.com/rate_limit`
- **Supabase Dashboard**: https://supabase.com/dashboard

---

## ✅ Checklist

Before deploying, ensure:

- [ ] Gemini API key is set and tested
- [ ] `.env.local` is in `.gitignore`
- [ ] GitHub token is set (if using private repos)
- [ ] Hugging Face token is set (for faster downloads)
- [ ] All keys are kept secret
- [ ] Keys are rotated regularly
- [ ] Billing alerts are configured
- [ ] Rate limits are understood
- [ ] Backup keys are available

---

**Need Help?**
- Check troubleshooting section above
- Review README.md for setup instructions
- Check backend logs: `tail -f backend.log`
- Test API keys manually before use
