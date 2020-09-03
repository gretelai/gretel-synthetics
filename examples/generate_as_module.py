
"""
Example module on how to run data generation from a standlone python invocation. Tensorflow
requires that processes are launch with "spawn" mode, for which it is a good practice to ensure
that any code is only executed after checking that we are in the main module
(`if __name__ == '__main__'`).

In the event that you choose to export a Notebook to a pure module, please note the changes below. These
changes will have a ``NOTE:`` comment.
"""

from pathlib import Path

from gretel_synthetics.config import LocalConfig
from gretel_synthetics.generate import generate_text

PARALLELISM = 0

# Create a config that we can use for both training and generating data
# The default values for ``max_lines`` and ``epochs`` are optimized for training on a GPU.


# NOTE: Update your ``checkpoint_dir`` and other config params as needed
config = LocalConfig(
    max_lines=0,          # maximum lines of training data. Set to ``0`` to train on entire file
    max_line_len=2048,    # the max line length for input training data
    epochs=15,            # 15-50 epochs with GPU for best performance
    vocab_size=20000,     # tokenizer model vocabulary size
    gen_lines=1000,       # the number of generated text lines
    dp=True,              # train with differential privacy enabled (privacy assurances, but reduced accuracy)
    field_delimiter=",",  # specify if the training text is structured, else ``None``
    overwrite=True,       # overwrite previously trained model checkpoints
    checkpoint_dir=(Path.cwd() / 'checkpoints').as_posix(),
    input_data_path="https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/uber_scooter_rides_1day.csv"
)


# Let's generate some text!
#
# The ``generate_text`` function is a generator that will return
# a line of predicted text based on the ``gen_lines`` setting in your
# config.
#
# There is no limit on the line length as with proper training, your model
# should learn where newlines generally occur. However, if you want to
# specify a maximum char len for each line, you may set the ``gen_chars``
# attribute in your config object


# Optionally, when generating text, you can provide a callable that takes the
# generated line as a single arg. If this function raises any errors, the
# line will fail validation and will not be returned.  The exception message
# will be provided as a ``explain`` field in the resulting dict that gets
# created by ``generate_text``
def validate_record(line):
    rec = line.split(", ")
    if len(rec) == 6:
        float(rec[5])
        float(rec[4])
        float(rec[3])
        float(rec[2])
        int(rec[0])
    else:
        raise Exception('record not 6 parts')


# NOTE: You should put the actual generation routine into a function, that can be
# called after the parent python processes is done bootstrapping
def start():
    for line in generate_text(config, line_validator=validate_record, parallelism=PARALLELISM):
        print(line)


# NOTE: It is preferred to always invoke your generation this way. Simply invoking start() from
# the top-level of the main module *should* work, but YMMV.
if __name__ == "__main__":
    start()
