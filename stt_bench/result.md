# From DE
backend      ok       wer      cer avg_final_ms p95_final_ms   cost_usd finals partials errors
----------------------------------------------------------------------------------------------
local        True    0.241    0.064       4942.6       6741.0          -     11        0      0
deepgram     True    0.139    0.041       2248.5       3410.0          -     12       44      0
openai       True    0.063    0.021      51298.0      51298.0          -      1      176      0
google       True    1.000    0.992      50714.0      50714.0          -      1      140      1
yandex       True    0.101    0.029      47960.0      47960.0          -      1      148      0
elevenlabs   True    1.051    1.010      12326.2      11753.0          -      4       49      0
speechmatics True    0.215    0.072        871.8       1161.0          -     74      102      2

# From RU
backend      ok       wer      cer avg_final_ms p95_final_ms   cost_usd finals partials errors
----------------------------------------------------------------------------------------------
local        True    0.241    0.064       5170.4       6948.0          -     11        0      0
deepgram     False    1.000    1.000            -            -          -      0        0      1
openai       False    1.000    1.000            -            -          -      0        0      0
google       False    1.000    1.000            -            -          -      0        0      0
yandex       True    0.101    0.029      47916.0      47916.0          -      1      148      0
elevenlabs   False    1.000    1.000            -            -          -      0        0      0
speechmatics True    0.215    0.072        865.9       1103.0          -     74      102      2