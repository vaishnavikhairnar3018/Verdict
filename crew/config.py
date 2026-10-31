import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("CREW_MODEL", "gemini-3.6-flash")
FALLBACK_MODEL = os.getenv("CREW_FALLBACK_MODEL", "gemini-3.5-flash-lite")   # used if MODEL is overloaded

QUERIES_PER_ROUND = 3     # how many web searches the Researcher runs per round
RESULTS_PER_QUERY = 4     # how many results we keep from each search
MAX_ROUNDS = 2            # hard stop so the loop can never run forever
MIN_SUPPORTED = 3         # stop early once we have this many trustworthy claims
KEEP_THRESHOLD = 0.6      # final confidence a claim needs to appear in the answer
