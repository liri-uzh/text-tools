#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this list is taken from
# http://raw.githubusercontent.com/gosia-malgosia/german-stop-words/refs/heads/master/german-stop-words.txt
# related blog post: https://medium.com/idealo-tech-blog/common-pitfalls-with-the-preprocessing-of-german-text-for-nlp-3cfb8dc19ebe
STOP_WORDS = (
    {
        "de": frozenset(
            [
                "aber",
                "alle",
                "allem",
                "allen",
                "aller",
                "alles",
                "als",
                "also",
                "am",
                "an",
                "ander",
                "andere",
                "anderem",
                "anderen",
                "anderer",
                "anderes",
                "anderm",
                "andern",
                "anders",
                "auch",
                "auf",
                "aus",
                "bei",
                "bin",
                "bis",
                "bist",
                "da",
                "damit",
                "dann",
                "das",
                "dass",
                "dasselbe",
                "dazu",
                "daß",
                "dein",
                "deine",
                "deinem",
                "deinen",
                "deiner",
                "deines",
                "dem",
                "demselben",
                "den",
                "denn",
                "denselben",
                "der",
                "derer",
                "derselbe",
                "derselben",
                "des",
                "desselben",
                "dessen",
                "dich",
                "die",
                "dies",
                "diese",
                "dieselbe",
                "dieselben",
                "diesem",
                "diesen",
                "dieser",
                "dieses",
                "dir",
                "doch",
                "dort",
                "du",
                "durch",
                "ein",
                "eine",
                "einem",
                "einen",
                "einer",
                "eines",
                "einig",
                "einige",
                "einigem",
                "einigen",
                "einiger",
                "einiges",
                "einmal",
                "er",
                "es",
                "etwas",
                "euch",
                "euer",
                "eure",
                "eurem",
                "euren",
                "eurer",
                "eures",
                "für",
                "gegen",
                "gewesen",
                "hab",
                "habe",
                "haben",
                "hat",
                "hatte",
                "hatten",
                "hier",
                "hin",
                "hinter",
                "ich",
                "ihm",
                "ihn",
                "ihnen",
                "ihr",
                "ihre",
                "ihrem",
                "ihren",
                "ihrer",
                "ihres",
                "im",
                "in",
                "indem",
                "ins",
                "ist",
                "jede",
                "jedem",
                "jeden",
                "jeder",
                "jedes",
                "jene",
                "jenem",
                "jenen",
                "jener",
                "jenes",
                "jetzt",
                "kann",
                "kein",
                "keine",
                "keinem",
                "keinen",
                "keiner",
                "keines",
                "können",
                "könnte",
                "machen",
                "man",
                "manche",
                "manchem",
                "manchen",
                "mancher",
                "manches",
                "mein",
                "meine",
                "meinem",
                "meinen",
                "meiner",
                "meines",
                "mich",
                "mir",
                "mit",
                "muss",
                "musste",
                "nach",
                "nicht",
                "nichts",
                "noch",
                "nun",
                "nur",
                "ob",
                "oder",
                "ohne",
                "sein",
                "seine",
                "seinem",
                "seinen",
                "seiner",
                "seines",
                "selbst",
                "sich",
                "sie",
                "sind",
                "so",
                "solche",
                "solchem",
                "solchen",
                "solcher",
                "solches",
                "soll",
                "sollte",
                "sondern",
                "sonst",
                "um",
                "und",
                "uns",
                "unse",
                "unsem",
                "unsen",
                "unser",
                "unses",
                "unter",
                "vom",
                "von",
                "vor",
                "war",
                "waren",
                "warst",
                "was",
                "weil",
                "weiter",
                "welche",
                "welchem",
                "welchen",
                "welcher",
                "welches",
                "wenn",
                "werde",
                "werden",
                "wie",
                "wieder",
                "will",
                "wir",
                "wird",
                "wirst",
                "wo",
                "wollen",
                "wollte",
                "während",
                "würde",
                "würden",
                "zu",
                "zum",
                "zur",
                "zwar",
                "zwischen",
                "über",
            ]
        ),
    },
)


CONNECTOR_WORDS = {
    "de": frozenset(
        [
            "ohne",
            "ein",
            "eine",
            "zu",
            "bei",
            "von",
            "und",
            "an",
            "in",
            "oder",
            "für",
            "aus",
            "auf",
            "der",
            "die",
            "das",
            "des",
            "dem",
            "den",
            "dass",
            "mit",
            "über",
            "durch",
            "nach",
            "gegen",
            "entlang",
            "unter",
            "um",
            "vor",
            "hinter",
            "zwischen",
            "während",
            "außer",
            "innerhalb",
            "nahe",
            "außerhalb",
            "ausserhalbaber",
            "noch",
            "so",
            "doch",
            "als",
            "weil",
            "wenn",
            "seit",
            "da",
            "dass",
            "obwohl",
            "es sei denn",
            "bis",
            "wo",
            "solange",
            "falls",
            "obgleich",
            "unterhalb",
            "neben",
            "jenseits",
            "dieser",
            "dieses",
            "diese",
            "jene",
            "solcher",
            "solches",
            "solche",
            "welcher",
            "welches",
            "welche",
            "wessen",
            "deren",
            "dessen",
        ]
    ),
}
