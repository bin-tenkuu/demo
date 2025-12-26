package demo.model.auth;

import demo.entity.SysUser;
import demo.model.JpaQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.Predicate;
import jakarta.persistence.criteria.Root;
import lombok.Getter;
import lombok.Setter;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/// @author bin
/// @since 2023/05/30
@Getter
@Setter
@Schema(description = "系统用户查询")
public class SysUserQuery extends JpaQuery<SysUser> {

    /// 用户昵称
    private String nickName;

    /// 用户邮箱
    private String email;

    /// 手机号码
    private String phonenumber;

    /// 帐号状态（0正常 1停用）
    private String status;

    /// 最后登录IP
    private String loginIp;

    /// 最后登录时间
    private LocalDateTime loginDateStart;

    /// 最后登录时间
    private LocalDateTime loginDateEnd;

    /// 创建时间
    private LocalDateTime createTimeStart;

    private LocalDateTime createTimeEnd;

    /// 更新时间
    private LocalDateTime updateTimeStart;

    /// 更新时间
    private LocalDateTime updateTimeEnd;

    @Override
    public List<Predicate> toPredicate(Root<SysUser> root, CriteriaBuilder cb) {
        var list = new ArrayList<Predicate>();
        if (StringUtils.hasLength(nickName)) {
            list.add(cb.like(root.get("nickName"), "%" + nickName + "%"));
        }
        if (StringUtils.hasLength(email)) {
            list.add(cb.like(root.get("email"), "%" + email + "%"));
        }
        if (StringUtils.hasLength(phonenumber)) {
            list.add(cb.like(root.get("phonenumber"), "%" + phonenumber + "%"));
        }
        if (StringUtils.hasLength(status)) {
            list.add(cb.equal(root.get("status"), status));
        }
        if (StringUtils.hasLength(loginIp)) {
            list.add(cb.like(root.get("loginIp"), "%" + loginIp + "%"));
        }
        if (loginDateStart != null) {
            list.add(cb.greaterThanOrEqualTo(root.get("loginDate"), loginDateStart));
        }
        if (loginDateEnd != null) {
            list.add(cb.lessThanOrEqualTo(root.get("loginDate"), loginDateEnd));
        }
        if (createTimeStart != null) {
            list.add(cb.greaterThanOrEqualTo(root.get("createTime"), createTimeStart));
        }
        if (createTimeEnd != null) {
            list.add(cb.lessThanOrEqualTo(root.get("createTime"), createTimeEnd));
        }
        if (updateTimeStart != null) {
            list.add(cb.greaterThanOrEqualTo(root.get("updateTime"), updateTimeStart));
        }
        if (updateTimeEnd != null) {
            list.add(cb.lessThanOrEqualTo(root.get("updateTime"), updateTimeEnd));
        }
        return list;
    }

}
