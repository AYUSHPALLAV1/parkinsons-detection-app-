map_dict = {
  'A':'𐌀', 'B':'𐌁', 'C':'𐌂', 'D':'𐌃', 'E':'𐌄', 'F':'𐌅', 'G':'Ᏽ', 'H':'𐋅', 'I':'𐌉', 'J':'𐌔', 'K':'𐌊', 'L':'𐌋', 'M':'𐌌', 'N':'𐌍', 'O':'Ꝋ', 'P':'𐌐', 'Q':'ᑫ', 'R':'𐌓', 'S':'𐌔', 'T':'𐌕', 'U':'𐌖', 'V':'𐌖', 'W':'Ꮤ', 'X':'𐌗', 'Y':'𐌙', 'Z':'𐌆',
  'a':'𐌀', 'b':'𐌁', 'c':'𐌂', 'd':'𐌃', 'e':'𐌄', 'f':'𐌅', 'g':'Ᏽ', 'h':'𐋅', 'i':'𐌉', 'j':'𐌔', 'k':'𐌊', 'l':'𐌋', 'm':'𐌌', 'n':'𐌍', 'o':'Ꝋ', 'p':'𐌐', 'q':'ᑫ', 'r':'𐌓', 's':'𐌔', 't':'𐌕', 'u':'𐌖', 'v':'𐌖', 'w':'Ꮤ', 'x':'𐌗', 'y':'𐌙', 'z':'𐌆'
}

texts = [
    "Eye Tracking heuristics dynamically measure saccades and gaze velocity via precise facial mesh frameworks.",
    "Advanced diagnostic engines isolate acoustic perturbation anomalies like jitter and shimmering vocal frequencies.",
    "Deep Neurological CNN analyzes digitized handwriting spiral patterns to identify critical motor micro-tremors."
]

with open("output.txt", "w", encoding="utf-8") as f:
    for t in texts:
        out = "".join(map_dict.get(c, c) for c in t)
        f.write(out + "\n")
