import json
import numpy as np
import nltk
from collections import Counter
import random
import os

nltk.download('punkt_tab', quiet=True)

random.seed(42)
np.random.seed(42)

class_0_base_templates = [
    "cannot login to {system}",
    "password reset link not working",
    "account locked after multiple failed attempts",
    "unable to access {system}",
    "forgot password and cannot reset",
    "login credentials not recognized",
    "two factor authentication not sending code",
    "single sign on not working",
    "cannot access {system}",
    "user account disabled",
    "email password expired",
    "cannot authenticate with {system}",
    "login page shows error message",
    "session timeout too frequent",
    "cannot access {system}",
    "vpn connection requires password reset",
    "account access denied",
    "cannot log into {system}",
    "authentication token expired",
    "biometric login not working",
    "smart card reader not recognized",
    "cannot access {system}",
    "oauth login failing",
    "ldap authentication error",
    "cannot access {system}",
    "password complexity requirements not met",
    "account suspended due to security",
    "cannot access {system}",
    "login redirects to error page",
    "cannot access {system}",
    "my {system} account is locked",
    "password expired for {system}",
    "cannot sign in to {system}",
    "authentication failed for {system}",
    "access denied to {system}",
    "login button not responding",
    "multi factor authentication not working",
    "cannot unlock my account",
    "password reset email not received",
    "account credentials invalid"
]

class_1_base_templates = [
    "{app} crashes on save",
    "error {code} when submitting form",
    "{app} freezes when opening large file",
    "database connection timeout error",
    "javascript error in web browser",
    "{app} crashes when running macro",
    "{app} shows blank screen",
    "button click does nothing",
    "data not saving correctly in {app}",
    "{app} slow and unresponsive",
    "error message appears randomly",
    "cannot export report to pdf from {app}",
    "{app} closes unexpectedly",
    "form validation not working",
    "search function returns no results in {app}",
    "{app} hangs on startup",
    "data corruption in saved files",
    "print preview shows blank page",
    "{app} cannot connect to server",
    "error when uploading large file to {app}",
    "{app} shows wrong data",
    "cannot import csv file into {app}",
    "{app} memory leak detected",
    "error when generating report in {app}",
    "{app} interface not loading",
    "cannot delete records from database",
    "{app} shows duplicate entries",
    "error when syncing data in {app}",
    "{app} cannot find configuration file",
    "validation error on correct input",
    "{app} keeps freezing",
    "error code {code} in {app}",
    "{app} not responding",
    "data export failing in {app}",
    "{app} installation corrupted",
    "plugin error in {app}",
    "{app} update broke functionality",
    "cannot open files in {app}",
    "{app} performance degraded"
]

class_2_base_templates = [
    "{device} overheating and shutting down",
    "{device} not responding since morning",
    "keyboard keys not working properly",
    "monitor screen flickering constantly",
    "mouse cursor jumping around screen",
    "hard drive making clicking sounds",
    "{device} battery not holding charge",
    "usb ports not detecting devices",
    "webcam not working during video calls",
    "speakers producing no sound",
    "{device} fan running at maximum speed",
    "external monitor not displaying",
    "keyboard backlight not turning on",
    "touchpad not responding to gestures",
    "{device} screen has dead pixels",
    "{device} paper jam error",
    "scanner not scanning documents",
    "headphones microphone not working",
    "{device} charger port loose",
    "desktop computer not powering on",
    "network adapter not connecting",
    "bluetooth not pairing with devices",
    "{device} hinge broken and loose",
    "{device} showing low ink warning",
    "external hard drive not mounting",
    "{device} keyboard keys stuck",
    "monitor showing no signal",
    "webcam light stuck on",
    "{device} trackpad clicking not working",
    "{device} printing blank pages",
    "{device} screen cracked",
    "{device} power button not working",
    "{device} speakers crackling",
    "{device} camera not focusing",
    "{device} microphone muted",
    "{device} volume control broken",
    "{device} display resolution wrong",
    "{device} keyboard layout incorrect",
    "{device} touchscreen not responding"
]

class_3_base_templates = [
    "wifi keeps disconnecting every {time} minutes",
    "cannot reach {system} site connection timeout",
    "internet connection very slow today",
    "vpn connection drops frequently",
    "email server not responding",
    "network drive not accessible",
    "cannot ping internal servers",
    "dns resolution failing",
    "firewall blocking legitimate traffic",
    "network printer offline",
    "voip phone system down",
    "{system} site not loading",
    "database server unreachable",
    "backup server connection failed",
    "network switch port not working",
    "wireless access point not broadcasting",
    "ethernet cable connection unstable",
    "network authentication server down",
    "proxy server timeout errors",
    "cannot access {system}",
    "network latency very high",
    "dhcp server not assigning ip addresses",
    "network file share permissions error",
    "remote desktop connection timeout",
    "network monitoring system offline",
    "cannot access external websites",
    "network bandwidth usage unusually high",
    "network security certificate expired",
    "network routing table corrupted",
    "cannot access network attached storage",
    "wifi signal very weak",
    "network adapter driver error",
    "cannot connect to {system}",
    "internet speed dropped significantly",
    "network configuration lost",
    "vpn authentication failed",
    "proxy settings incorrect",
    "network card not detected",
    "wireless network not found"
]

class_4_base_templates = [
    "request installation of {software} on my {device}",
    "need access to {system}",
    "request software license for {software}",
    "need additional storage space on {system}",
    "request access to {system}",
    "need new user account for {role}",
    "request upgrade to latest {software} version",
    "need access to {system}",
    "request installation of {software}",
    "need access to {system}",
    "request new email distribution list",
    "need access to {system}",
    "request installation of {software}",
    "need access to {system}",
    "request new shared folder for {team}",
    "need access to {system}",
    "request installation of {software}",
    "need access to {system}",
    "request new virtual machine for {purpose}",
    "need access to {system}",
    "request installation of {software}",
    "need access to {system}",
    "request new {system} workspace for project",
    "need access to {system}",
    "request installation of {software}",
    "need access to {system}",
    "request new service account for {purpose}",
    "need access to {system}",
    "request installation of {software}",
    "need access to {system}",
    "request permission to {action}",
    "need {resource} for my project",
    "request {resource} access",
    "need to install {software}",
    "request {system} account creation"
]

systems = ["email", "vpn", "crm", "erp", "intranet", "sharepoint", "cloud storage", 
           "company portal", "shared drive", "project management tool", "time tracking system",
           "internal wiki", "remote desktop", "active directory", "cloud services",
           "api documentation portal", "code repository", "monitoring dashboard",
           "ticketing system", "analytics platform", "customer database", 
           "financial reporting system", "hr management system", "marketing automation tool",
           "document management system", "business intelligence tool"]

apps = ["crm application", "excel", "word", "outlook", "chrome", "firefox", "safari",
        "powerpoint", "access", "project", "visio", "teams", "slack", "zoom",
        "photoshop", "illustrator", "autocad", "solidworks", "matlab", "python",
        "vscode", "eclipse", "intellij", "docker", "postgresql", "mysql", "oracle"]

devices = ["laptop", "desktop", "printer", "scanner", "tablet", "phone", "monitor",
           "keyboard", "mouse", "webcam", "speakers", "headphones", "projector"]

software = ["vscode", "python development tools", "docker desktop", "git client",
            "nodejs and npm", "postgresql database", "jupyter notebook", "photoshop",
            "office", "visual studio", "eclipse", "intellij", "matlab", "autocad"]

roles = ["intern", "new employee", "contractor", "consultant", "temporary worker"]

teams = ["engineering", "marketing", "sales", "hr", "finance", "operations", "support"]

purposes = ["testing", "development", "staging", "production", "training", "demo"]

actions = ["access production database", "deploy to server", "modify system settings",
           "install software", "create user accounts", "modify network configuration"]

resources = ["server", "database", "api key", "ssl certificate", "domain", "ip address"]

error_codes = ["500", "404", "403", "401", "502", "503", "504", "400", "408"]

time_intervals = ["5", "10", "15", "30", "60"]

synonyms_map = {
    "cannot": ["unable to", "can't", "cannot", "not able to", "failed to"],
    "not": ["not", "is not", "doesn't", "isn't", "won't"],
    "login": ["login", "sign in", "log in", "access account", "authenticate"],
    "error": ["error", "issue", "problem", "failure", "bug"],
    "request": ["request", "need", "require", "would like", "asking for"],
    "access": ["access", "use", "open", "reach", "connect to"],
    "not working": ["not working", "broken", "failing", "down", "malfunctioning"],
    "help": ["help", "assistance", "support", "guidance"],
    "installation": ["installation", "install", "setup", "deployment"],
    "crash": ["crash", "freeze", "hang", "stop responding", "close unexpectedly"]
}

time_phrases = [
    "", " since yesterday", " since this morning", " since last week",
    " for the past hour", " all day", " since the update", " recently",
    " since this afternoon", " for the last few days", " since monday",
    " since the weekend", " for several hours now"
]

urgency_phrases = [
    "", " please help", " urgent", " asap", " need assistance",
    " please fix", " high priority", " blocking my work",
    " critical issue", " please respond quickly", " time sensitive"
]

intro_phrases = [
    "", "having issue with ", "problem: ", "issue: ",
    "hi, ", "hello, ", "i need help with ", "ticket: ",
    "good morning, ", "good afternoon, ", "i'm experiencing ",
    "there's a problem with ", "i cannot ", "help needed: "
]

filler_words = ["suddenly", "now", "again", "still", "always", "sometimes", "recently"]

typo_chars = {
    "a": ["a", "e"],
    "e": ["e", "a", "i"],
    "i": ["i", "e"],
    "o": ["o", "u"],
    "u": ["u", "o"],
    "l": ["l", "i"],
    "r": ["r", "l"],
    "n": ["n", "m"],
    "m": ["m", "n"]
}

def add_typos(text, prob=0.12):
    if random.random() > prob:
        return text
    words = text.split()
    if not words:
        return text
    word_idx = random.randint(0, len(words) - 1)
    word = words[word_idx]
    if len(word) > 3:
        char_idx = random.randint(1, len(word) - 2)
        char = word[char_idx].lower()
        if char in typo_chars:
            replacement = random.choice(typo_chars[char])
            word = word[:char_idx] + replacement + word[char_idx + 1:]
            words[word_idx] = word
    return " ".join(words)

def replace_placeholders(template):
    if "{system}" in template:
        template = template.replace("{system}", random.choice(systems), 1)
    if "{app}" in template:
        template = template.replace("{app}", random.choice(apps), 1)
    if "{device}" in template:
        template = template.replace("{device}", random.choice(devices), 1)
    if "{software}" in template:
        template = template.replace("{software}", random.choice(software), 1)
    if "{role}" in template:
        template = template.replace("{role}", random.choice(roles), 1)
    if "{team}" in template:
        template = template.replace("{team}", random.choice(teams), 1)
    if "{purpose}" in template:
        template = template.replace("{purpose}", random.choice(purposes), 1)
    if "{action}" in template:
        template = template.replace("{action}", random.choice(actions), 1)
    if "{resource}" in template:
        template = template.replace("{resource}", random.choice(resources), 1)
    if "{code}" in template:
        template = template.replace("{code}", random.choice(error_codes), 1)
    if "{time}" in template:
        template = template.replace("{time}", random.choice(time_intervals), 1)
    return template

def apply_synonyms(text):
    result = text
    for word, replacements in synonyms_map.items():
        if word in result.lower():
            if random.random() < 0.4:
                replacement = random.choice(replacements)
                result = result.replace(word, replacement, 1)
    return result

def create_variation(base_text):
    result = base_text
    
    variation_ops = []
    
    if random.random() < 0.6:
        variation_ops.append("synonym")
    if random.random() < 0.5:
        variation_ops.append("intro")
    if random.random() < 0.4:
        variation_ops.append("time")
    if random.random() < 0.35:
        variation_ops.append("urgency")
    if random.random() < 0.2:
        variation_ops.append("filler")
    
    if "synonym" in variation_ops:
        result = apply_synonyms(result)
    
    if "intro" in variation_ops:
        intro = random.choice(intro_phrases)
        if intro:
            result = intro + result
    
    if "time" in variation_ops:
        time_phrase = random.choice(time_phrases)
        if time_phrase:
            result = result + time_phrase
    
    if "urgency" in variation_ops:
        urgency = random.choice(urgency_phrases)
        if urgency:
            result = result + urgency
    
    if "filler" in variation_ops and random.random() < 0.5:
        filler = random.choice(filler_words)
        words = result.split()
        if len(words) > 2:
            insert_pos = random.randint(1, len(words) - 1)
            words.insert(insert_pos, filler)
            result = " ".join(words)
    
    if random.random() < 0.18:
        result = add_typos(result, prob=0.25)
    
    return result

def generate_unique_samples(target_per_class=800, max_attempts_per_class=5000):
    unique_samples = set()
    texts = []
    labels = []
    
    all_templates = [
        (class_0_base_templates, 0),
        (class_1_base_templates, 1),
        (class_2_base_templates, 2),
        (class_3_base_templates, 3),
        (class_4_base_templates, 4)
    ]
    
    for templates, label in all_templates:
        class_samples = 0
        attempts = 0
        
        print(f"Generating samples for class {label}...")
        
        while class_samples < target_per_class and attempts < max_attempts_per_class:
            base_template = random.choice(templates)
            filled_template = replace_placeholders(base_template)
            variation = create_variation(filled_template)
            normalized_text = variation.strip().lower()
            
            key = (normalized_text, int(label))
            
            if key not in unique_samples:
                unique_samples.add(key)
                texts.append(normalized_text)
                labels.append(label)
                class_samples += 1
            
            attempts += 1
        
        if class_samples < target_per_class:
            print(f"  Warning: Only generated {class_samples} unique samples for class {label} after {attempts} attempts")
        else:
            print(f"  Generated {class_samples} unique samples for class {label}")
    
    return texts, labels

texts, labels = generate_unique_samples(target_per_class=800, max_attempts_per_class=5000)

print(f"\nTotal unique samples generated: {len(texts)}")

tokenized_texts = []
for text in texts:
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    if tokens:
        tokenized_texts.append(tokens)
    else:
        tokenized_texts.append(["unknown"])

all_tokens = []
for tokens in tokenized_texts:
    all_tokens.extend(tokens)

word_counts = Counter(all_tokens)
words = sorted(word_counts.keys())

os.makedirs("data", exist_ok=True)

words_json = json.dumps(words, indent=2)
with open("data/words.json", 'w') as f:
    f.write(words_json)

text_json = json.dumps(tokenized_texts, indent=2)
with open("data/text.json", 'w') as f:
    f.write(text_json)

labels_array = np.array(labels, dtype=np.int64)
np.save('data/labels.npy', labels_array)

print(f"\nVocabulary size: {len(words)} words.")
print("Class distribution:")
for i in range(5):
    count = sum(1 for l in labels if l == i)
    print(f"  Class {i}: {count} samples")
print("\nSaved data/words.json, data/text.json, data/labels.npy.")
print(f"\nUniqueness check: {len(texts)} unique (text, label) pairs generated.")
