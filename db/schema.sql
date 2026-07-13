CREATE TABLE IF NOT EXISTS `matches` (
  `analysis_id` VARCHAR(128) NOT NULL,
  `matches_video_id` VARCHAR(128) NULL,
  `user_id` VARCHAR(128) NULL,
  `source_url` TEXT NULL,
  `output_dir` VARCHAR(1024) NULL,
  `roster_file` VARCHAR(1024) NULL,
  `status` VARCHAR(32) NOT NULL DEFAULT 'finished',
  `roster_json` JSON NULL,
  `player_stats_json` JSON NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`analysis_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `roster_players` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `analysis_id` VARCHAR(128) NOT NULL,
  `team_name` VARCHAR(128) NOT NULL,
  `team_color` VARCHAR(128) NULL,
  `jersey_number` INT NOT NULL,
  `roster_player_json` JSON NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uniq_roster_player` (`analysis_id`, `team_name`, `jersey_number`),
  CONSTRAINT `fk_roster_players_match`
    FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `roster_known_stats` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `analysis_id` VARCHAR(128) NOT NULL,
  `metric` VARCHAR(128) NOT NULL,
  `team_name` VARCHAR(128) NULL,
  `stat_value_json` JSON NULL,
  `known_stats_json` JSON NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uniq_roster_known_stat` (`analysis_id`, `metric`, `team_name`),
  CONSTRAINT `fk_roster_known_stats_match`
    FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `ai_player_stats` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `analysis_id` VARCHAR(128) NOT NULL,
  `player_key` VARCHAR(128) NOT NULL,
  `jersey_number` INT NULL,
  `team_name` VARCHAR(128) NULL,
  `player_name` VARCHAR(255) NULL,
  `verification_status` VARCHAR(64) NULL,
  `stats_json` JSON NULL,
  `player_json` JSON NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uniq_ai_player_stats` (`analysis_id`, `player_key`),
  CONSTRAINT `fk_ai_player_stats_match`
    FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `events` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `analysis_id` VARCHAR(128) NOT NULL,
  `event_index` INT NOT NULL,
  `event_type` VARCHAR(64) NULL,
  `frame` INT NULL,
  `time_s` DOUBLE NULL,
  `primary_player` VARCHAR(128) NULL,
  `secondary_player` VARCHAR(128) NULL,
  `confidence` DOUBLE NULL,
  `identity_confidence` DOUBLE NULL,
  `identity_confidence_receiver` DOUBLE NULL,
  `status` VARCHAR(64) NULL,
  `event_json` JSON NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uniq_event` (`analysis_id`, `event_index`),
  CONSTRAINT `fk_events_match`
    FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `clips` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `analysis_id` VARCHAR(128) NOT NULL,
  `clip_index` INT NOT NULL,
  `clip_path` VARCHAR(1024) NOT NULL,
  `event_type` VARCHAR(64) NULL,
  `player` VARCHAR(128) NULL,
  `player_boxed` BOOLEAN NOT NULL DEFAULT FALSE,
  `upload_status` VARCHAR(64) NOT NULL DEFAULT 'pending',
  `time_s` DOUBLE NULL,
  `event_frame` INT NULL,
  `size_bytes` BIGINT NULL,
  `clip_json` JSON NOT NULL,
  `upload_json` JSON NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uniq_clip` (`analysis_id`, `clip_index`),
  CONSTRAINT `fk_clips_match`
    FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
