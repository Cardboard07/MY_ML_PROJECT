import logging

logging.basicConfig(
    level=logging.INFO,   # change to DEBUG when needed
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename="run.log",
    filemode="w"
)

logger = logging.getLogger(__name__)