
CREATE TABLE season (
    season_id VARCHAR(20) NOT NULL PRIMARY KEY,
    year SMALLINT NOT NULL CHECK (year BETWEEN 1900 AND 2025) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL
);

CREATE TABLE team (
    team_id VARCHAR(20) NOT NULL PRIMARY KEY,
    team_name VARCHAR(255) UNIQUE NOT NULL,
    coach_name VARCHAR(255) NOT NULL,
    region VARCHAR(20) UNIQUE NOT NULL
);

CREATE TABLE player (
    player_id VARCHAR(20) NOT NULL PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    dob DATE CHECK (dob < '2016-01-01') NOT NULL,
    batting_hand VARCHAR(20) CHECK (batting_hand IN ('Right', 'Left')) NOT NULL,
    bowling_skill VARCHAR(20) CHECK (bowling_skill IN ('fast', 'medium', 'legspin', 'offspin')) NOT NULL,
    country_name VARCHAR(50)
);

CREATE TABLE auction (
    auction_id VARCHAR(20) NOT NULL PRIMARY KEY,
    season_id VARCHAR(20) NOT NULL REFERENCES season(season_id),
    player_id VARCHAR(20) NOT NULL REFERENCES player(player_id),
    base_price BIGINT CHECK (base_price >= 1000000) NOT NULL,
    sold_price BIGINT,
    is_sold BOOLEAN NOT NULL DEFAULT FALSE,
    team_id VARCHAR(20) REFERENCES team(team_id),
    UNIQUE (player_id, team_id, season_id),
    
    CONSTRAINT check_sold_conditions CHECK (
        (is_sold = FALSE)
        OR
        (is_sold = TRUE AND sold_price IS NOT NULL AND team_id IS NOT NULL AND sold_price >= base_price)
    )
);

CREATE OR REPLACE FUNCTION validate_auction() 
RETURNS TRIGGER AS $$
BEGIN
   
    IF NEW.is_sold = TRUE THEN
        IF NEW.sold_price IS NULL OR NEW.team_id IS NULL THEN
            RAISE EXCEPTION 'Violation: **null** constraint - sold_price and team_id cannot be null when is_sold is true';
        END IF;
        
        IF NEW.sold_price < NEW.base_price THEN
            RAISE EXCEPTION 'Violation: **null** constraint - sold_price must be greater than or equal to base_price';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER check_auction_constraints
    BEFORE INSERT OR UPDATE ON auction
    FOR EACH ROW
    EXECUTE FUNCTION validate_auction();

CREATE TABLE match (
    match_id VARCHAR(20) NOT NULL PRIMARY KEY,
    match_type VARCHAR(20) NOT NULL CHECK (match_type IN ('league', 'playoff', 'knockout')),
    venue VARCHAR(20) NOT NULL REFERENCES team(region),
    team_1_id VARCHAR(20) NOT NULL REFERENCES team(team_id),
    team_2_id VARCHAR(20) NOT NULL REFERENCES team(team_id),
    match_date DATE NOT NULL,
    season_id VARCHAR(20) NOT NULL REFERENCES season(season_id),
    win_run_margin SMALLINT,
    win_by_wickets SMALLINT,
    win_type VARCHAR(20) CHECK (win_type IN ('runs', 'wickets', 'draw')),
    toss_winner SMALLINT CHECK (toss_winner IN (1, 2)),
    toss_decide VARCHAR(20) CHECK (toss_decide IN ('bowl' , 'bat')),
    winner_team_id VARCHAR(20) REFERENCES team(team_id),
    
    CONSTRAINT check_win_type_constraints CHECK (
        (win_type = 'draw' AND win_run_margin IS NULL AND win_by_wickets IS NULL AND winner_team_id IS NULL)
        OR
        (win_type != 'draw' AND (
            (win_run_margin IS NOT NULL AND win_by_wickets IS NULL)
            OR
            (win_run_margin IS NULL AND win_by_wickets IS NOT NULL)
        ))
        OR
        (win_type = 'runs' AND win_by_wickets IS NULL AND (
            (toss_decide = 'bat' AND winner_team_id = team_1_id)
            OR
            (toss_decide = 'bat' AND winner_team_id = team_2_id)
        ))
        OR
        (win_type = 'wickets' AND win_run_margin IS NULL AND (
            (toss_decide = 'bowl' AND winner_team_id = team_1_id)
            OR
            (toss_decide = 'bowl' AND winner_team_id = team_2_id)
        ))
    )
);


CREATE TABLE balls (
    match_id VARCHAR(20) NOT NULL REFERENCES match(match_id),
    innings_num SMALLINT NOT NULL,
    over_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    striker_id VARCHAR(20) NOT NULL REFERENCES player(player_id),
    non_striker_id VARCHAR(20) NOT NULL REFERENCES player(player_id),
    bowler_id VARCHAR(20) NOT NULL REFERENCES player(player_id),
    PRIMARY KEY (match_id, innings_num, over_num, ball_num)
);

CREATE TABLE batter_score (
    match_id VARCHAR(20) NOT NULL REFERENCES match(match_id),
    over_num SMALLINT NOT NULL,
    innings_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    run_scored SMALLINT NOT NULL CHECK (run_scored >= 0),
    type_run VARCHAR(20) CHECK (type_run IN ('running', 'boundary')),
    PRIMARY KEY (match_id, over_num, innings_num, ball_num),
   
    CONSTRAINT fk_balls FOREIGN KEY (match_id, innings_num, over_num, ball_num)
        REFERENCES balls (match_id, innings_num, over_num, ball_num)
);

CREATE TABLE extras (
    match_id VARCHAR(20) NOT NULL REFERENCES match(match_id),
    innings_num SMALLINT NOT NULL,
    over_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    extra_runs SMALLINT NOT NULL CHECK (extra_runs >= 0),
    extra_type VARCHAR(20) NOT NULL CHECK (extra_type IN ('no_ball', 'wide', 'byes', 'legbyes')),
    PRIMARY KEY (match_id, innings_num, over_num, ball_num),

    CONSTRAINT fk_balls FOREIGN KEY (match_id, innings_num, over_num, ball_num)
        REFERENCES balls (match_id, innings_num, over_num, ball_num)
);

CREATE TABLE wickets (
    match_id VARCHAR(20) NOT NULL REFERENCES match(match_id),
    innings_num SMALLINT NOT NULL,
    over_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    player_out_id VARCHAR(20) NOT NULL REFERENCES player(player_id),
    kind_out VARCHAR(20) NOT NULL CHECK (kind_out IN ('bowled', 'caught', 'lbw', 'runout', 'stumped', 'hitwicket')),
    fielder_id VARCHAR(20) REFERENCES player(player_id),
    PRIMARY KEY (match_id, innings_num, over_num, ball_num),
    
    CONSTRAINT fk_balls FOREIGN KEY (match_id, innings_num, over_num, ball_num)
        REFERENCES balls (match_id, innings_num, over_num, ball_num),

    CONSTRAINT check_fielder_id CHECK (
        (kind_out IN ('caught', 'runout', 'stumped') AND fielder_id IS NOT NULL)
        OR kind_out NOT IN ('caught', 'runout', 'stumped')
    )
);

-- Function to check if fielder is a wicketkeeper for 'stumped' outs
CREATE OR REPLACE FUNCTION check_stumped_wicketkeeper()
RETURNS TRIGGER AS $$
BEGIN
    
    IF NEW.kind_out = 'stumped' THEN
    
        IF NOT EXISTS (
            SELECT 1
            FROM player_match
            WHERE player_match.player_id = NEW.fielder_id
            AND player_match.role = 'wicketkeeper'
        ) THEN
            RAISE EXCEPTION 'Fielder must be a wicketkeeper for stumped out!';
        END IF;
    END IF;

    -- Ensure fielder_id is not null for 'caught', 'runout', or 'stumped'
    IF NEW.kind_out IN ('caught', 'runout', 'stumped') AND NEW.fielder_id IS NULL THEN
        RAISE EXCEPTION 'Fielder_id cannot be NULL for "caught", "runout", or "stumped" outs!';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER trigger_check_stumped_wicketkeeper
BEFORE INSERT OR UPDATE ON wickets
FOR EACH ROW EXECUTE FUNCTION check_stumped_wicketkeeper();



CREATE TABLE player_match (
    player_id VARCHAR(20) NOT NULL REFERENCES player(player_id),
    match_id VARCHAR(20) NOT NULL REFERENCES match(match_id),
    role VARCHAR(20) NOT NULL CHECK (role IN ('batter', 'bowler', 'allrounder', 'wicketkeeper')),
    team_id VARCHAR(20) NOT NULL REFERENCES team(team_id),
    is_extra BOOLEAN NOT NULL,
    PRIMARY KEY (player_id, match_id)
);

CREATE TABLE player_team (
    player_id VARCHAR(20) NOT NULL,
    team_id VARCHAR(20) NOT NULL,
    season_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (player_id, team_id, season_id),

    CONSTRAINT fk_auction FOREIGN KEY (player_id, team_id, season_id)
        REFERENCES auction(player_id, team_id, season_id)
);

CREATE TABLE awards (
    match_id VARCHAR(20) NOT NULL REFERENCES match(match_id),
    award_type VARCHAR(20) NOT NULL CHECK (award_type IN ('orange_cap', 'purple_cap')),
    player_id VARCHAR(20) NOT NULL REFERENCES player(player_id),
    PRIMARY KEY (match_id, award_type)
);

CREATE OR REPLACE FUNCTION insert_player_team() 
RETURNS TRIGGER AS $$
BEGIN

    IF NEW.is_sold THEN
        INSERT INTO player_team (player_id, team_id, season_id)
        VALUES (NEW.player_id, NEW.team_id, NEW.season_id);
    END IF;
    RETURN NEW;  
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER after_auction_insert
AFTER INSERT ON auction
FOR EACH ROW 
EXECUTE FUNCTION insert_player_team();





CREATE OR REPLACE FUNCTION generate_season_id() 
RETURNS TRIGGER AS $$
BEGIN
    NEW.season_id := 'IPL' || NEW.year;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER before_season_insert
BEFORE INSERT ON season
FOR EACH ROW EXECUTE FUNCTION generate_season_id();

CREATE OR REPLACE FUNCTION validate_match_id() 
RETURNS TRIGGER AS $$
DECLARE
    match_count INT;
    match_serial INT;
BEGIN

    IF NEW.match_id NOT LIKE NEW.season_id || '___' THEN
        RAISE EXCEPTION 'match_id format is invalid';
    END IF;

    match_serial := CAST(SUBSTRING(NEW.match_id FROM LENGTH(NEW.season_id) + 1 FOR 3) AS INT);

    IF match_serial < 1 OR match_serial > 999 THEN
        RAISE EXCEPTION 'match_id serial number out of range (must be between 001 and 999)';
    END IF;
    SELECT COUNT(*) INTO match_count
    FROM match
    WHERE season_id = NEW.season_id AND match_id = NEW.match_id;

    IF match_count > 0 THEN
        RAISE EXCEPTION 'match_id already exists for the given season_id';
    END IF;

    SELECT COUNT(*) INTO match_count
    FROM match
    WHERE season_id = NEW.season_id;

    IF match_count >= 999 THEN
        RAISE EXCEPTION 'sequence of match id violated: maximum limit reached for this season';
    END IF;
    NEW.match_id := NEW.season_id || LPAD((match_count + 1)::text, 3, '0');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER before_match_insert
BEFORE INSERT OR UPDATE ON match
FOR EACH ROW EXECUTE FUNCTION validate_match_id();


-- Trigger to limit international players

CREATE OR REPLACE FUNCTION limit_international_players() 
RETURNS TRIGGER AS $$
DECLARE
    international_count INT;
BEGIN
    SELECT COUNT(*) INTO international_count
    FROM player_team pt
    JOIN player p ON pt.player_id = p.player_id
    WHERE pt.team_id = NEW.team_id AND pt.season_id = NEW.season_id
    AND p.country_name <> 'India';
    IF international_count >= 3 THEN
        RAISE EXCEPTION 'there could be at most 3 international players per team per season';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER before_player_team_insert
BEFORE INSERT ON player_team
FOR EACH ROW EXECUTE FUNCTION limit_international_players();


------------------------------ Trigger to limit home matches ----------------------------------------

CREATE OR REPLACE FUNCTION limit_home_matches() RETURNS TRIGGER AS $$
DECLARE
    home_match_count_team_1 INT;
    home_match_count_team_2 INT;
BEGIN
    IF NEW.match_type = 'league' THEN
        IF NEW.venue <> (SELECT region FROM team WHERE team_id = NEW.team_1_id)
           AND NEW.venue <> (SELECT region FROM team WHERE team_id = NEW.team_2_id) THEN
            RAISE EXCEPTION 'League match must be played at home ground of one of the teams';
        END IF;
        SELECT COUNT(*) INTO home_match_count_team_1
        FROM match
        WHERE season_id = NEW.season_id
          AND match_type = 'league'
          AND team_1_id = NEW.team_1_id
          AND team_2_id = NEW.team_2_id
          AND venue = (SELECT region FROM team WHERE team_id = NEW.team_1_id);

        SELECT COUNT(*) INTO home_match_count_team_2
        FROM match
        WHERE season_id = NEW.season_id
          AND match_type = 'league'
          AND team_1_id = NEW.team_2_id
          AND team_2_id = NEW.team_1_id
          AND venue = (SELECT region FROM team WHERE team_id = NEW.team_2_id);
        IF home_match_count_team_1 >= 1 OR home_match_count_team_2 >= 1 THEN
            RAISE EXCEPTION 'Each team can play only one home match in a league against another team';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER before_match_insert_or_update
BEFORE INSERT OR UPDATE ON match
FOR EACH ROW EXECUTE FUNCTION limit_home_matches();


----------------------------- updating rows ---------------------------------------------

CREATE OR REPLACE FUNCTION update_match_results() RETURNS TRIGGER AS $$
DECLARE
    orange_cap_player VARCHAR(20);
    purple_cap_player VARCHAR(20);
BEGIN
    -- Update winner_team_id
    IF NEW.win_type = 'draw' THEN
        NEW.winner_team_id := NULL;
    ELSE
        NEW.winner_team_id := CASE
            WHEN NEW.win_type = 'runs' THEN NEW.team_1_id
            WHEN NEW.win_type = 'wickets' THEN NEW.team_2_id
            END;
    END IF;

    -- Insert awards
    SELECT player_id INTO orange_cap_player
    FROM batter_score
    WHERE match_id = NEW.match_id
    GROUP BY player_id
    ORDER BY SUM(run_scored) DESC, player_id
    LIMIT 1;

    INSERT INTO awards (match_id, award_type, player_id)
    VALUES (NEW.match_id, 'orange_cap', orange_cap_player);

    SELECT player_out_id INTO purple_cap_player
    FROM wickets
    WHERE match_id = NEW.match_id
    GROUP BY player_out_id
    ORDER BY COUNT(*) DESC, player_out_id
    LIMIT 1;

    INSERT INTO awards (match_id, award_type, player_id)
    VALUES (NEW.match_id, 'purple_cap', purple_cap_player);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER after_match_update
AFTER UPDATE ON match
FOR EACH ROW 
EXECUTE FUNCTION update_match_results();


------------------------------------- Deletion --------------------------------------

CREATE OR REPLACE FUNCTION delete_auction_related_data() 
RETURNS TRIGGER AS $$
BEGIN

    DELETE FROM player_team
    WHERE player_id = OLD.player_id;

    DELETE FROM awards
    WHERE player_id = OLD.player_id;

    DELETE FROM player_match
    WHERE player_id = OLD.player_id;

    DELETE FROM batter_score
    WHERE player_id = OLD.player_id;

    DELETE FROM balls
    WHERE player_id = OLD.player_id;

    DELETE FROM extras
    WHERE player_id = OLD.player_id;

    DELETE FROM wickets
    WHERE player_id = OLD.player_id;

    DELETE FROM auction
    WHERE player_id = OLD.player_id AND is_sold = true;

    RETURN OLD;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER trigger_delete_auction
AFTER DELETE ON auction
FOR EACH ROW
EXECUTE FUNCTION delete_auction_related_data();



CREATE OR REPLACE FUNCTION delete_match_related_data() 
RETURNS TRIGGER AS $$
BEGIN

    DELETE FROM awards
    WHERE match_id = OLD.match_id;

    
    DELETE FROM extras
    WHERE match_id = OLD.match_id;


    DELETE FROM batter_score
    WHERE match_id = OLD.match_id;

    
    DELETE FROM wickets
    WHERE match_id = OLD.match_id;

    
    DELETE FROM player_match
    WHERE match_id = OLD.match_id;

    DELETE FROM balls
    WHERE match_id = OLD.match_id;

    RETURN OLD;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER trigger_delete_match
AFTER DELETE ON match
FOR EACH ROW
EXECUTE FUNCTION delete_match_related_data();


CREATE OR REPLACE FUNCTION delete_season_related_data() 
RETURNS TRIGGER AS $$
BEGIN
    
    DELETE FROM auction
    WHERE season_id = OLD.season_id;

    
    DELETE FROM awards
    WHERE season_id = OLD.season_id;

    
    DELETE FROM balls
    WHERE match_id IN (SELECT match_id FROM match WHERE season_id = OLD.season_id);

    
    DELETE FROM batter_score
    WHERE match_id IN (SELECT match_id FROM match WHERE season_id = OLD.season_id);

    
    DELETE FROM extras
    WHERE match_id IN (SELECT match_id FROM match WHERE season_id = OLD.season_id);


    DELETE FROM match
    WHERE season_id = OLD.season_id;


    DELETE FROM player_match
    WHERE match_id IN (SELECT match_id FROM match WHERE season_id = OLD.season_id);


    DELETE FROM player_team
    WHERE season_id = OLD.season_id;


    DELETE FROM wickets
    WHERE match_id IN (SELECT match_id FROM match WHERE season_id = OLD.season_id);

    RETURN OLD;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER trigger_delete_season
AFTER DELETE ON season
FOR EACH ROW
EXECUTE FUNCTION delete_season_related_data();


--------------------------------- Views ----------------------------------------


CREATE VIEW batter_stats AS
SELECT
    stats.player_id,
    COUNT(DISTINCT stats.match_id) AS Mat, 
    COUNT(DISTINCT stats.innings_num) AS Inns,  
    SUM(stats.run_scored) + COALESCE(SUM(stats.extra_runs), 0) AS R,  
    MAX(stats.run_scored) AS HS,  
    CASE 
        WHEN COUNT(DISTINCT stats.innings_num) = 0 THEN 0 
        -- ELSE (SUM(stats.run_scored) + COALESCE(SUM(stats.extra_runs), 0)) / COUNT(DISTINCT stats.innings_num) 
        ELSE (SUM(stats.run_scored)) / COUNT(DISTINCT stats.innings_num) 
    END AS Avg,  
    CASE 
        -- WHEN (SUM(stats.run_scored) + COALESCE(SUM(stats.extra_runs), 0)) = 0 THEN 0 
        -- ELSE ((SUM(stats.run_scored) + COALESCE(SUM(stats.extra_runs), 0)) / NULLIF(COUNT(DISTINCT stats.ball_num), 0)) * 100 
        WHEN (SUM(stats.run_scored)) = 0 THEN 0 
        ELSE ((SUM(stats.run_scored)) / NULLIF(COUNT(DISTINCT stats.ball_num), 0)) * 100 
    END AS SR, 
    COUNT(CASE WHEN stats.run_scored >= 100 THEN 1 END) AS "100s",
    COUNT(CASE WHEN stats.run_scored >= 50 AND stats.run_scored < 100 THEN 1 END) AS "50s", 
    COUNT(CASE WHEN stats.run_scored = 0 THEN 1 END) AS Ducks,  
    COUNT(DISTINCT stats.ball_num) AS BF, 
    COUNT(CASE WHEN stats.type_run = 'boundary' THEN 1 END) AS Boundaries,  
    COUNT(CASE WHEN stats.is_not_out THEN 1 END) AS NO  
FROM (
    SELECT
        b.striker_id AS player_id,
        b.match_id,
        b.innings_num,
        SUM(bs.run_scored) AS run_scored,
        bs.type_run, 
        b.ball_num,
        SUM(e.extra_runs) AS extra_runs,
        CASE WHEN SUM(bs.run_scored) > 0 THEN FALSE ELSE TRUE END AS is_not_out  
    FROM balls b
    LEFT JOIN batter_score bs ON b.match_id = bs.match_id AND b.innings_num = bs.innings_num AND b.ball_num = bs.ball_num  -- Join batter_score to get runs
    LEFT JOIN extras e ON b.match_id = e.match_id AND b.innings_num = e.innings_num AND b.over_num = e.over_num AND b.ball_num = e.ball_num  -- Join extras to exclude no-balls/wides
    WHERE b.striker_id IS NOT NULL  
    AND NOT EXISTS (
        SELECT 1
        FROM extras e2
        WHERE e2.match_id = b.match_id
          AND e2.innings_num = b.innings_num
          AND e2.over_num = b.over_num
          AND e2.ball_num = b.ball_num  
    )
    GROUP BY b.striker_id, b.match_id, b.innings_num, b.ball_num, bs.type_run  
) AS stats
GROUP BY stats.player_id;  


CREATE VIEW bowler_stats AS
SELECT
    b.bowler_id AS player_id,  
    COUNT(DISTINCT b.ball_num) AS B,  
    COUNT(DISTINCT CASE WHEN w.kind_out IN ('bowled', 'caught', 'lbw', 'stumped') THEN w.player_out_id END) AS W,  
    SUM(bs.run_scored) + COALESCE(SUM(e.extra_runs), 0) AS Runs,  
    
    CASE 
        WHEN COUNT(DISTINCT CASE WHEN w.kind_out IN ('bowled', 'caught', 'lbw', 'stumped') THEN w.player_out_id END) = 0 THEN 0
        ELSE (SUM(bs.run_scored) + COALESCE(SUM(e.extra_runs), 0)) / COUNT(DISTINCT CASE WHEN w.kind_out IN ('bowled', 'caught', 'lbw', 'stumped') THEN w.player_out_id END)
    END AS Avg,

    CASE 
        WHEN COUNT(DISTINCT b.ball_num) = 0 THEN 0
        ELSE (SUM(bs.run_scored) + COALESCE(SUM(e.extra_runs), 0)) / (COUNT(DISTINCT b.ball_num) / 6.0)  -- Economy rate
    END AS Econ,
    
    CASE 
        WHEN COUNT(DISTINCT CASE WHEN w.kind_out IN ('bowled', 'caught', 'lbw', 'stumped') THEN w.player_out_id END) = 0 THEN 0
        ELSE COUNT(DISTINCT b.ball_num) / COUNT(DISTINCT CASE WHEN w.kind_out IN ('bowled', 'caught', 'lbw', 'stumped') THEN w.player_out_id END)  -- Strike rate
    END AS SR,

    COALESCE(SUM(e.extra_runs), 0) AS Extras
    
FROM balls b
LEFT JOIN wickets w 
    ON b.match_id = w.match_id 
    AND b.innings_num = w.innings_num 
    AND b.over_num = w.over_num 
    AND b.ball_num = w.ball_num  
    
LEFT JOIN batter_score bs 
    ON b.match_id = bs.match_id 
    AND b.innings_num = bs.innings_num 
    AND b.over_num = bs.over_num 
    AND b.ball_num = bs.ball_num  
    
LEFT JOIN extras e 
    ON b.match_id = e.match_id 
    AND b.innings_num = e.innings_num 
    AND b.over_num = e.over_num 
    AND b.ball_num = e.ball_num  
    
WHERE b.bowler_id IS NOT NULL 
GROUP BY b.bowler_id;  -


CREATE VIEW fielder_stats AS
SELECT
    fielder_id as player_id,
    COUNT(CASE WHEN kind_out = 'caught' THEN 1 END) AS C,
    COUNT(CASE WHEN kind_out = 'stumped' THEN 1 END) AS St,
    COUNT(CASE WHEN kind_out = 'runout' THEN 1 END) AS RO
FROM wickets
GROUP BY player_id;
