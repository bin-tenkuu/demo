package demo.core.event;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationListener;
import org.springframework.stereotype.Component;
import org.springframework.web.context.support.ServletRequestHandledEvent;

/**
 * @author bin
 * @since 2026/02/09
 */
@Slf4j
@Component
public class AfterServletRequest implements ApplicationListener<ServletRequestHandledEvent> {
    @Override
    public void onApplicationEvent(ServletRequestHandledEvent event) {
        log.info("请求完成: {} {} {}ms {}",
                event.getMethod(), event.getStatusCode(), event.getProcessingTimeMillis(), event.getRequestUrl());
    }

}
