import os
from pathlib import Path
from gretel_synthetics.config import LocalConfig
from gretel_synthetics.generator import _load_model
import tensorflow as tf
from random import shuffle

def predict_sequence_det(model: tf.keras.models.Sequential, seq_len: int, batch_size: int, total_lines: int):
    def generate_predictions(input_eval, selection):
        all_predictions = tf.constant([[0] for _ in range(batch_size)], dtype=tf.int32)

        for _ in range(seq_len):
            prediction = model(input_eval)[:, -1, :]
            _, top_indices = tf.math.top_k(prediction, total_lines)
            input_eval = tf.gather(top_indices, selection, axis=1, batch_dims=1)
            all_predictions = tf.concat([all_predictions, input_eval], axis=1)

        return all_predictions

    init_eval = tf.constant([[6, 3] for _ in range(batch_size)])
    return generate_predictions, init_eval


def test_deterministic_batch_predict():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    golden = None

    for batch_size in (1,4,16):
        config = LocalConfig(
            field_delimiter=",",  # specify if the training text is structured, else ``None``
            checkpoint_dir=(Path.cwd() / '../test-notebooks/checkpoints/batch_0').as_posix(),
            input_data_path='dummy',
            predict_batch_size=batch_size,
        )

        _, model = _load_model(config)

        for compile in (False, True):
            predict, init_eval = predict_sequence_det(model, 10, batch_size, 16)
            if compile:
                predict = tf.function(predict)

            selection = [i for i in range(16)]

            for _ in range(5):
                shuffle(selection)
                print(selection)

                sequences = [[] for _ in range(16)]
                # sequences[i]: sequence predicted by always selecting i most likeliest next token

                # model.reset_states() -- this will break if uncommented and call in l.59 is commented out
                for ofs in range(0, 16, batch_size):
                    sub_selection = selection[ofs:ofs+batch_size]
                    selection_tensor = tf.constant([[idx] for idx in sub_selection])
                    print(selection_tensor)

                    model.reset_states()
                    predictions = predict(init_eval, selection_tensor)
                    for i, seq_idx in enumerate(sub_selection):
                        sequences[seq_idx] = [int(x.numpy()) for x in predictions[i, :]]

                print('Predicted:', str(sequences))
                if golden is None:
                    golden = sequences
                else:
                    assert sequences == golden



