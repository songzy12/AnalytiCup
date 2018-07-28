english_train_path = '../input/cikm_english_train_20180516.txt'
spanish_train_path = '../input/cikm_spanish_train_20180516.txt'
unlabel_spanish_train_path = '../input/cikm_unlabel_spanish_train_20180516.txt'
test_path = '../input/cikm_test_a_20180516.txt'

en_vec_path = '../input/wiki.en.vec'
es_vec_path = '../input/wiki.es.vec'

embed_size = 300
max_features = 5500
max_features = 4300  # this is for only es dataset
maxlen = 54

es_token_list = ['set_ratio', 'sort_ratio', 'no', 'ha', 'la', 'con', 'el', 'te', 'es', 'de', 'mi', 'o', 'los', 'hay', 'para', 'entre', 'tengo', 'otra',
                 '?', 'cuando', 'lo', 'como', 'una', 'mis', 'esta', 'este', 'ti', 'eran', 'qué', 'he', 'esto', 'y', 'que', 'estoy', '.', '¿', 'me', 'ni', 'en', 'a']

es_5w1h_list = ['cuándo', 'por qué', 'qué', 'como', 'puedo']