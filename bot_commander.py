from main import (MODEL_NAME, build_seq2seq_model, init_inference,
                  initial_setup, load_model_weights, load_vocabulary)

CORPUS_NAME = 'twitter'

# We're only interested in the metadata object
metadata = initial_setup(CORPUS_NAME)[0]

(word2idx, idx2word, unk_id, pad_id, start_id,
 end_id, vocabulary_size) = load_vocabulary(metadata)

model_ = build_seq2seq_model(vocabulary_size)
load_model_weights(model_)


seeds = ["I want to eat burger", "come home", "hillary had nothing to do",
         "only reason i can exercise is because i can breathe now", "i never heard of it until you mentioned it"]

while True:
    print("Type quit for exist the prompt")
    seed = input(">")

    if seed == 'quit':
        break

    print("Query >", seed)

    inference = init_inference(model_, word2idx, idx2word, unk_id, start_id)
    top_n = 3
    for i in range(top_n):
        sentence = inference(seed, top_n)
        print(" >", ' '.join(sentence))
