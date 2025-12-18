package demo.events;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;

/**
 * @author bin
 * @since 2025/12/18
 */
@Slf4j
// @Component
public class AfterCommandLine implements CommandLineRunner {
    @Override
    public void run(String... args) {
        if (args == null || args.length == 0) {
            return;
        }
        log.info("has args: ({})", String.join(", ", args));
    }
}
