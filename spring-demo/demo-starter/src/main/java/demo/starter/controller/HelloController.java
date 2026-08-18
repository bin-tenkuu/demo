package demo.starter.controller;

import demo.starter.exchange.ExchangeApi;
import demo.starter.retrofit.TestApi;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/11
 */
@Tag(name = "hello")
@RestController
@RequestMapping
@RequiredArgsConstructor
public class HelloController {
    private final ExchangeApi exchangeApi;
    private final TestApi testApi;
    private int n = 0;

    @GetMapping("/hello")
    public String hello() {
        n++;
        if (n % 3 != 0) {
            throw new RuntimeException("test exception");
        }
        return "Hello, Spring Boot!";
    }

    @GetMapping("/hello/exchange")
    public String helloExchange() {
        return exchangeApi.hello();
    }

    @GetMapping("/hello/retrofit")
    public String helloRetrofit() {
        return testApi.hello();
    }

}
