
if __name__ == "__main__":
    log_file_name = get_log_file_name()
    logger = setup_logger(log_file_name, logging.CRITICAL)
    main_human_vs_ai()