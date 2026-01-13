from sentence_transformers import SentenceTransformer, util

from config import MODEL_DIR
from utils.check_cuda import check_cuda_and_gpus
from utils.string_norm import strip_accents_and_lowercase

xsent = [
    "οι δε φαρισεοι ακουσαντες οτι εφιμωσε τους σαδδουκεους συνηχθησαν επι το αυτο",  # 6580
    "ειπεν δε αυτοις οταν προσευχησθε λεγετε πατερ αγιασθητω το ονομα σου ελθατω η βασιλια σου γενηθητω το θελημα σου ως εν ουρανω και επι γης",  # 4348
    "νυν κρισις εστιν του κοσμου νυν ο αρχων τουτου τουτου νυν ο αρχων του κοσμου τουτου εκβληθησεται εξω",  # 3478
    "διο προσλαμβανεσθαι αλληλους καθως και ο χς προσελαβετο υμας εις δοξαν του θυ",  # 7872
]

ysent = [
    "οι δε φαρισαιοι ακουσαντες οτι εφιμωσε τους σαδδουκεους συνηχθησαν επι το αυτο",  # 6580
    "ειπεν δε αυτοις οταν προσευχησθε λεγετε πατερ αγιασθητω το ονομα σου ελθατω η βασιλια σου γενηθητω το θελημα σου ως εν ουρανω και επι γης και ρυσαι ημας απο του πονηρου",  # 4348
    "νυν δε προς σε ερχομαι και ταυτα λαλω εν τω κοσμω ινα εχωσιν την χαραν την εμην πεπληρωκενην εν αυτοις",  # 3637
    "μετανοησαται ουν και επιστρεψαται προς το εξαλιφθηναι υμων τας αμαρτιας",  # 2115
]

xsent_norm = [strip_accents_and_lowercase(s) for s in xsent]
ysent_norm = [strip_accents_and_lowercase(s) for s in ysent]

device = check_cuda_and_gpus()
print(f"Device: {device}")

model = SentenceTransformer(MODEL_DIR, device=device)

x_embeddings = model.encode(xsent_norm, convert_to_tensor=True)
y_embeddings = model.encode(ysent_norm, convert_to_tensor=True)

print("Similarities:")
for i in range(len(xsent_norm)):
    similarity = util.cos_sim(x_embeddings[i], y_embeddings[i]).item()
    print(f"Pair {i + 1}: {similarity:.4f}")
