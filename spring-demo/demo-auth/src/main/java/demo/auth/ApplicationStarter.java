package demo.auth;

import org.jetbrains.annotations.NotNull;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationListener;
import org.springframework.context.event.ContextRefreshedEvent;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/11
 */
@SpringBootApplication
public class ApplicationStarter implements ApplicationListener<ContextRefreshedEvent> {
    public static void main(String[] args) {
        SpringApplication.run(ApplicationStarter.class, args);
    }

    @Override
    public void onApplicationEvent(@NotNull ContextRefreshedEvent event) {
        System.out.println("启动成功");
    }

}
