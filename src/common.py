"""Shared text-cleaning constants and helpers."""
import re, html
from better_profanity import profanity

# NLTK's English stopword list, inlined (the corpus download is blocked in this sandbox).
STOP = set("""i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its itself they them
their theirs themselves what which who whom this that that'll these those am is are was were be
been being have has had having do does did doing a an the and but if or because as until while of
at by for with about against between into through during before after above below to from up down
in out on off over under again further then once here there when where why how all any both each
few more most other some such no nor not only own same so than too very s t can will just don
don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn
doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn
needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't""".split())

SYMPTOMS = ["sore","fever","fatigue","malaise","muscle pain","headache","throat","vomiting",
            "diarrhoea","diarrhea","abdominal pain","rash","kidney","liver","ill","weak",
            "platelets","rest","sick","fighting","numb","flu"]

def clean_tweet(t):
    if isinstance(t, float): return ""
    r = t.lower()
    r = profanity.censor(r)
    r = html.unescape(r)
    r = re.sub(r"@[A-Za-z0-9_]+", "", r)
    r = re.sub(r"#[A-Za-z0-9_]+", "", r)
    r = re.sub(r"http\S+", "", r)
    r = re.sub(r"[()!?]", " ", r)
    r = re.sub(r"\[.*?\]", " ", r)
    r = re.sub(r"\'", "", r)
    r = re.sub(r"[^a-z0-9]", " ", r)
    r = re.sub(r"\brt\b", "", r)
    r = [w for w in r.split() if w not in STOP and len(w) > 2]
    return " ".join(r)

