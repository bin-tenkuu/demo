create table sys_user
(
    id           integer not null
        primary key autoincrement,
    nick_name    text    not null,
    email        text,
    phone_number text,
    avatar       text,
    status       integer  default 0,
    login_ip     text,
    login_date   text,
    create_by    text,
    create_time  datetime default current_timestamp,
    update_by    text,
    update_time  datetime default current_timestamp
);
create table sys_user_auth
(
    username text    not null,
    type     text    not null,
    user_id  integer not null,
    password text    not null,
    primary key (username),
    unique (type, user_id)
);
create table sys_role
(
    id          integer not null
        primary key autoincrement,
    role_key    text,
    role_name   text,
    role_sort   integer,
    status      integer,
    create_by   text,
    create_time datetime default current_timestamp,
    update_by   text,
    update_time datetime default current_timestamp
);
create table sys_menu
(
    id          integer not null
        primary key autoincrement,
    menu_name   text,
    parent_id   integer,
    order_num   integer,
    path        text,
    component   text,
    query       text,
    is_frame    integer,
    is_cache    integer,
    menu_type   text,
    visible     integer,
    status      integer,
    perms       text,
    icon        text,
    create_by   text,
    create_time datetime default current_timestamp,
    update_by   text,
    update_time datetime default current_timestamp,
    unique (menu_name)
);
create table sys_user_role
(
    user_id integer not null,
    role_id integer not null,
    primary key (user_id, role_id)
);
create table sys_role_menu
(
    role_id integer not null,
    menu_id integer not null,
    primary key (role_id, menu_id)
);
