package demo.starter.exchange;

import org.springframework.web.service.annotation.GetExchange;
import org.springframework.web.service.annotation.HttpExchange;

/**
 * @author bin
 * @since 2026/08/18
 */
@HttpExchange
public interface ExchangeApi {
    @GetExchange("/hello")
    String hello();
}
