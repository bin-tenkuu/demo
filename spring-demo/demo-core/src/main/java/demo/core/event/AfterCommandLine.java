package demo.core.event;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

/**
 * @author bin
 * @since 2025/12/18
 */
@Slf4j
@Component
public class AfterCommandLine implements CommandLineRunner {
    @Override
    public void run(String... args) {
        if (args.length == 0) {
            return;
        }
        log.info("has args: ({})", String.join(", ", args));
    }

}
