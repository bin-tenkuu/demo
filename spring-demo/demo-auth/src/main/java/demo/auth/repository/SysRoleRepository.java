package demo.auth.repository;


import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import demo.auth.entity.SysRole;
import demo.auth.mapper.SysRoleMapper;
import org.springframework.stereotype.Service;

/// 角色信息表 服务实现类
///
/// @author bin
/// @since 2023-05-30 09:40:18
@Service
public class SysRoleRepository extends ServiceImpl<SysRoleMapper, SysRole> {

    public <E extends IPage<SysRole>> E page(E page, Long menuId, Wrapper<SysRole> queryWrapper) {
        return baseMapper.selectPageWithMenu(page, menuId, queryWrapper);
    }

    /// 校验角色权限是否唯一
    public boolean checkRoleKeyExist(SysRole role) {
        var roles = baseMapper.listByRoleKey(role.getRoleKey());
        for (var sysRole : roles) {
            if (!sysRole.getId().equals(role.getId())) {
                return true;
            }
        }
        return false;
    }

}
