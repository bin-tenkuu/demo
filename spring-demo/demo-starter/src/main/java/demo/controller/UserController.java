package demo.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.conditions.query.LambdaQueryChainWrapper;
import demo.entity.User;
import demo.entity.UserData;
import demo.mapper.UserDataMapper;
import demo.mapper.UserMapper;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * @author bin
 * @since 2025/05/06
 */
@Tag(name = "user")
@RestController
@RequestMapping
@RequiredArgsConstructor
public class UserController implements InitializingBean {
    private final UserMapper userMapper;
    private final UserDataMapper userDataMapper;

    @Override
    public void afterPropertiesSet() {
        userMapper.initTable();
        userDataMapper.initTable();
    }

    @GetMapping("/list")
    public List<User> list() {
        return userMapper.selectList(new QueryWrapper<>());
    }

    @PostMapping("/save")
    public void save(@RequestBody User user) {
        userMapper.insert(user);
    }

    @PostMapping("/update")
    public void update(@RequestBody User user) {
        userMapper.updateById(user);
        val wrapper = new LambdaQueryChainWrapper<>(userMapper).ge(User::getId, user.getId());
        val list = wrapper.list();

    }

    @GetMapping("/delete")
    public void delete(Integer id) {
        userMapper.deleteById(id);
    }

    @GetMapping("/data/list")
    public List<UserData> dataList(String id, LocalDateTime start, LocalDateTime end) {
        return userDataMapper.listBySnAndTimeRange(id, start, end, List.of("*"));
    }

    @PostMapping("/data/save")
    public void dataSave(@RequestBody UserData user) {
        userDataMapper.insert(user);
    }

    @PostMapping("/data/update")
    public void dataUpdate(@RequestBody UserData user) {
        userDataMapper.merge(user);
    }

    @GetMapping("/data/find")
    public UserData dataFind(LocalDateTime time, String id) {
        return userDataMapper.findById(time, id);
    }

}
