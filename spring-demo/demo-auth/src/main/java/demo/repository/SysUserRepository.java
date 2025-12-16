package demo.repository;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import demo.entity.SysUser;
import demo.mapper.SysUserMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

/**
 * @author bin
 * @since 2025/07/15
 */
@Service
@RequiredArgsConstructor
public class SysUserRepository extends ServiceImpl<SysUserMapper, SysUser> {

    public <E extends IPage<SysUser>> E page(E page, Long roleId, Wrapper<SysUser> queryWrapper) {
        return baseMapper.selectPageWithRole(page, roleId, queryWrapper);
    }

    public <E extends IPage<SysUser>> E pageUnAlloced(E page, Wrapper<SysUser> queryWrapper) {
        return baseMapper.selectPageWithoutRole(page,  queryWrapper);
    }

}
