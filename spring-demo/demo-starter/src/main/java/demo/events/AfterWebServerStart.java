package demo.events;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.web.servlet.context.ServletWebServerInitializedEvent;
import org.springframework.context.ApplicationListener;

import java.awt.*;
import java.net.URI;

/**
 * @author bin
 * @since 2025/12/18
 */
@Slf4j
// @Component
public class AfterWebServerStart implements ApplicationListener<ServletWebServerInitializedEvent> {
    @Override
    public void onApplicationEvent(ServletWebServerInitializedEvent event) {
        var webServer = event.getWebServer();
        var port = webServer.getPort();
        log.info("""
                        \n----------------------------------------
                        Application is running!
                        \tLocal:\t\thttp://127.0.0.1:{}
                        \tExternal:\thttp://0.0.0.0:{}
                        ----------------------------------------""",
                port, port);
        try {
            System.setProperty("java.awt.headless", "false");
            var desktop = Desktop.getDesktop();
            desktop.browse(new URI("http://127.0.0.1:" + port));
        } catch (Exception e) {
            log.info("Failed to open browser automatically.", e);
        }
    }
}
