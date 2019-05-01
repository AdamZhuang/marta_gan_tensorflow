import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def info(message, *message_args, **message_kwargs):
  logging.info(message, *message_args, **message_kwargs)
