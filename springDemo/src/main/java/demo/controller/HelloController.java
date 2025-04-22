package demo.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import demo.entity.User;
import demo.mapper.UserMapper;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/11
 */
@Tag(name = "hello")
@RestController
@RequestMapping
@RequiredArgsConstructor
public class HelloController implements InitializingBean {
    private final UserMapper userMapper;

    @Override
    public void afterPropertiesSet() {
        userMapper.initTable();
    }

    @GetMapping("/hello")
    public String hello() {
        return "Hello, Spring Boot!";
    }

    @GetMapping("/list")
    public List<User> list() {
        return userMapper.selectList(new QueryWrapper<>());
    }

    @PostMapping("/save")
    public void save(User user) {
        userMapper.insert(user);
    }

    @PostMapping("/update")
    public void update(User user) {
        userMapper.updateById(user);
    }

    @GetMapping("/delete")
    public void delete(Integer id) {
        userMapper.deleteById(id);
    }

}
