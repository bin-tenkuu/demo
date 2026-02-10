package demo.config;

import lombok.extern.slf4j.Slf4j;
import org.apache.coyote.AbstractProtocol;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.tomcat.TomcatWebServer;
import org.springframework.boot.web.server.context.WebServerInitializedEvent;
import org.springframework.context.ApplicationListener;
import org.springframework.stereotype.Component;

import java.awt.*;
import java.net.URI;

/**
 * @author bin
 * @since 2025/12/18
 */
@Slf4j
@Component
public class AfterWebServerStart implements ApplicationListener<WebServerInitializedEvent> {
    @Value("${server.open-browser:false}")
    private Boolean openBrowser;
    @Value("${server.servlet.context-path:/}")
    private String contextPath;

    @Override
    public void onApplicationEvent(WebServerInitializedEvent event) {
        var webServer = event.getWebServer();
        // 获取WebServer的实现类，这里假设是TomcatWebServer
        if (webServer instanceof TomcatWebServer tomcatWebServer) {
            // 获取Tomcat的Service
            var server = tomcatWebServer.getTomcat().getServer();
            // 获取Service中的所有Connector
            for (var service : server.findServices()) {
                for (var connector : service.findConnectors()) {
                    if (connector.getProtocolHandler() instanceof AbstractProtocol<?> ph) {
                        var address = ph.getAddress();
                        var scheme = connector.getScheme();
                        String ip = address == null ? "0.0.0.0" : address.getHostAddress();
                        var port = ph.getLocalPort();
                        System.out.printf("""
                                ----------------------------------------
                                Application is running!
                                \tLocal:\t\t%s://127.0.0.1:%s%s
                                \tExternal:\t%s://%s:%s%s
                                ----------------------------------------
                                """, scheme, port, contextPath, scheme, ip, port, contextPath);
                        if (openBrowser) {
                            openBrowser(scheme + "://127.0.0.1:" + port + contextPath);
                        }
                    }
                }
            }
        }
    }

    private static void openBrowser(String url) {
        try {
            System.setProperty("java.awt.headless", "false");
            var desktop = Desktop.getDesktop();
            desktop.browse(new URI(url));
        } catch (Exception e) {
            log.info("Failed to open browser automatically.", e);
        }
    }
}
