
    def get_log_file_name():
        log_num = 1
        while os.path.exists(f"log{log_num}.txt"):
            log_num += 1
        return f"log{log_num}.txt"

    def setup_logger(log_file, level=logging.INFO):
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger