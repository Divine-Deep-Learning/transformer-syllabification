# transformer-syllabification


We aim at showing that the usage of Transformer language model can be employed in Italian poetry text generation. We show two models, one that syllabifies and inserts caesura in any Italian hendecasyllable, and one that generates syllabified text with caesura in Dante's Divine Comedy style.
This repository contains the entire code that syllabifies only

# Demos of syllabified unseen hendecasyllables
(special tokens for space and verse ending are omitted for readability purposes)

```
INPUT:  le donne i cavallier l' arme gli amori  
PRED:   |le |don|ne i |ca|val|lier<c> |l' ar|me |gli a|mo|ri

INPUT:  le cortesie l' audaci imprese io canto  
PRED:   |le |cor|te|sie<c> |l' au|da|ci im|pre|se io |can|to
  
INPUT:  che furo al tempo che passaro i mori  
PRED:   |che |fu|ro al |tem|po<c> |che |pas|sa|ro i |mo|ri
  
INPUT:  d' africa il mare e in francia nocquer tanto  
PRED:   |d' af|ri|ca il |ma|re e<c> |in |fran|cia |noc|quer tan|to
```
